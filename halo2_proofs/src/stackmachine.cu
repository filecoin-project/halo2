#include <stdio.h>
#include <stdint.h>

// One instruction is 40 bytes long.
#define INSTRUCTION_SIZE 40

typedef struct {
  uint8_t capacity;
  // Number of items pushed so far. Eempty stack == 0.
  uint8_t top;
  //FIELD** items;
  FIELD* items;
} Stack;

DEVICE Stack stack_new(uint8_t capacity) {
  //FIELD* items = (FIELD*)malloc(sizeof(FIELD*) * capacity);
  FIELD* items = (FIELD*)malloc(sizeof(FIELD) * capacity);
  //printf("vmx: sizeof(FIELD*): %d", sizeof(FIELD*));
  //printf("vmx: sizeof(FIELD**): %d", sizeof(FIELD**));
  Stack stack = {.capacity = capacity, .top = 0, .items = items};
  return stack;
}

DEVICE void stack_push(Stack* stack, FIELD* item) {
  //stack->items[stack->top] = *item;
  memcpy(&stack->items[stack->top], item, sizeof(FIELD));
  stack->top += 1;
}

DEVICE FIELD stack_pop(Stack* stack) {
  FIELD item = stack->items[stack->top - 1];
  stack->top -= 1;
  return item;
}

typedef enum {
  /// Pops two elements, adds them and pushes the result.
  ADD = 1,
  /// Pops two elements, multiplies them and pushes the result.
  MUL,
  /// Pops one element, scales it and pushes the result.
  SCALE,
  /// Pushes one element.
  PUSH,
  /// Does some calculations and pushes the result. The position and omega is
  /// passed into the
  /// stack machine.
  LINEAR_TERM,
  /// Pushes the field element at `[poly_index][result-of-the-call]`;
  ROTATED,
} InstructionType;

typedef struct {
  InstructionType type;
  union {
    // For `ADD` and `MUL`.
    // Use a 64-bit type to make sure that the union is aligned to 8 byte
    // boundaries, independent of the types that follow. This is done as the
    // limb size of a field element may be 32 or 64-bit, hence the alignment
    // would differ.
    uint64_t none;
    // For `SCALE`, ` PUSH` and `LINEAR_TERM`.
    FIELD element;
    // For `ROTATED`.
    struct {
      uint32_t index;
      int32_t rotation;
    };
  };
} Instruction;

DEVICE void FIELD_print(FIELD a) {
  printf("0x");
  uint8_t* data = (uint8_t*)&a.val;
  for (uint8_t i = 0; i < 32; i++) {
    printf("%02x", data[i]);
  }
}

DEVICE uint get_rotated_pos(uint32_t pos, int32_t rotation, uint32_t poly_len) {
    int32_t new_pos = pos + rotation;

    // The position is at the beginning, the rotation is negative and so large, that it would lead
    // to an out of bounds error.
    if (new_pos < 0) {
        // Hence wrap around and use a position at the end of the polynomial.
        return poly_len + new_pos;
    }
    // The position is at the end, the rotation is positive and so large, that it would lead to an
    // out of bounds error.
    else if (new_pos > poly_len) {
        // Hence wrap around and use a position at the beginning of the polynomial.
        return new_pos - poly_len;
    }
    // It is outside those range, hence the rotation (being positive or negative) won't lead to an
    // out of bounds position.
    else {
        return new_pos;
    }
}

DEVICE void evaluate_at_pos(GLOBAL FIELD* polys, uint32_t poly_len, GLOBAL Instruction* instructions, uint32_t num_instructions, uint32_t stack_size, GLOBAL FIELD* omega, uint32_t pos, GLOBAL FIELD* result) {
// uint num_instructions = sizeof(instructions) / sizeof(Instruction);
// printf("vmx: sizeof instructions: %llu\n", sizeof(instructions));
// printf("vmx: sizeof Instruction: %llu\n", sizeof(Instruction));
  printf("vmx: num_instructions: %d\n", num_instructions);

// FIELD* stack = (FIELD *)malloc(sizeof(FIELD) * stack_size);
// FIELD* stack = malloc(stack_size * sizeof *FIELD);
  Stack stack = stack_new(stack_size);

// stack_push(&stack, omega);
// stack_pop(&stack);

 // Make it a 2-dimensional array for easier indexing.
// FIELD (*polys)[poly_len] = polys;

// int (*pointer)[poly_len];
// printf("vmx: polys[1][2] with pointer access: %x", (*(*(&polys + 1) + 2)).val[0]);
// printf("vmx: polys[1][2] with index   access: %d", polys[1][2]);

  for (uint i = 0; i < num_instructions; i++) {
    Instruction instruction = instructions[i];
    switch (instruction.type) {
      case ADD: {
        //printf("Add\n");
        FIELD lhs = stack_pop(&stack);
        FIELD rhs = stack_pop(&stack);
        stack_push(&stack, &FIELD_add(lhs, rhs));
        break;
      }
      case MUL: {
        //printf("Mul\n");
        FIELD lhs = stack_pop(&stack);
        FIELD rhs = stack_pop(&stack);
        stack_push(&stack, &FIELD_mul(lhs, rhs));
        break;
      }
      case SCALE: {
        //printf("Scale { scalar: }\n");
        FIELD lhs = stack_pop(&stack);
        stack_push(&stack, &FIELD_mul(lhs, instruction.element));
        break;
      }
      case PUSH: {
        //printf("Push { element: ");
        //FIELD_print(instruction.element);
        //printf(" }\n");
        stack_push(&stack, &instruction.element);
        break;
      }
      case LINEAR_TERM: {
        //printf("LinearTerm\n");
        //stack_push(&stack, &FIELD_mul(FIELD_pow(*omega, pos), instruction.element));
        FIELD* linear_term =  &FIELD_mul(FIELD_pow(*omega, pos), instruction.element);
        //printf("LinearTerm: ");
        //FIELD_print(*linear_term);
        //printf("\n");
        stack_push(&stack, linear_term);
        break;
      }
      case ROTATED: {
        //printf("Rotated { index: %d, rotation: %d }\n", instruction.index, instruction.rotation);
        uint32_t rotated_pos = get_rotated_pos(pos, instruction.rotation, poly_len);
        //printf("rotated pos: %d\n", rotated_pos);
        // `polys` is a two-dimensional array, but we cannot use it as usual two-dimensional array
        // as we don't know it's size at compile time, hence do some pointer arithmetic fun.
        FIELD* rotated = polys + (instruction.index * poly_len) + rotated_pos;
        //printf("rotated: ");
        //FIELD_print(*rotated);
        //printf("\n");
        stack_push(&stack, rotated);
        break;
      }
      default:
        printf("error: unknown instruction type!\n");
        break;
     }
  }
  *result = stack_pop(&stack);
}

// `poly_len` is the lengths of a single polynomial (all have the same length).
//KERNEL void evaluate(GLOBAL FIELD polys[][POLY_LEN], GLOBAL FIELD* result, uint32_t poly_len) {
//KERNEL void evaluate(GLOBAL FIELD polys[][], GLOBAL Instruction[] instructions, GLOBAL FIELD* omega, uint32_t poly_len, GLOBAL FIELD* result) {
KERNEL void evaluate(GLOBAL FIELD *polys, uint32_t polys_len, GLOBAL Instruction* instructions, uint32_t num_instructions, uint32_t stack_size, GLOBAL FIELD* omega, GLOBAL FIELD* result) {
    const uint32_t index = GET_GLOBAL_ID();

    if (index > 0) {
        return;
    }

    // TODO vmx 2022-10-22: Add the stride to common.cl in ec-gpu-gen and add an OpenCL version.
    const uint stride = blockDim.x * gridDim.x;

    evaluate_at_pos(polys, polys_len, instructions, num_instructions, stack_size, omega, 0, &result[0]);
    //for (uint32_t i = index; i < poly_len; i += stride) {
    //   // TODO vmx 2022-11-11: check if this if statement is really needed.
    //   if (i <= poly_len) {
    //       evaluate_at_pos(polys, instructions, omega, poly_len, i, &result[i]);
    //   }
    //}
}
