#include <stdint.h>
#include <stdio.h>

// TODO vmx 2022-12-15: This is kind of a hack. We define a maximum stack size,
// so that we don't need to allocate or manually use shared memory. The proper
// solution would be to use shared memory instead and use the acccess pattern
// as describeb in "Dynamic Indexing with Non-Uniform Access":
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
#define MAX_STACK_SIZE 10

typedef struct {
  uint8_t capacity;
  // Number of items pushed so far. Empty stack == 0.
  uint8_t top;
  FIELD items[MAX_STACK_SIZE];
} Stack;

DEVICE Stack stack_new(uint8_t capacity) {
  assert(capacity <= MAX_STACK_SIZE);
  Stack stack = {.capacity = capacity, .top = 0};
  return stack;
}

DEVICE void stack_push(Stack* stack, const FIELD item) {
  assert(stack->top < stack->capacity);
  stack->items[stack->top] = item;
  stack->top += 1;
}

DEVICE FIELD stack_pop(Stack* stack) {
  assert(stack->top > 0);
  FIELD item = stack->items[(stack->top - 1)];
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

DEVICE uint32_t get_rotated_pos(uint32_t pos,
                                int32_t rotation,
                                uint32_t poly_len) {
  // The relative position may be negative, it would then mean a position
  // counted from the back.
  int32_t rel_pos = pos + rotation;
  uint32_t new_pos;

  // The position is at the beginning, the rotation is negative and so large,
  // that it would lead to an out of bounds error.
  if (rel_pos < 0) {
    // Hence wrap around and use a position at the end of the polynomial.
    new_pos = poly_len + rel_pos;
  }
  // The position is at the end, the rotation is positive and so large, that it
  // would lead to an out of bounds error.
  else if (rel_pos >= poly_len) {
    // Hence wrap around and use a position at the beginning of the
    // polynomial.
    new_pos = rel_pos - poly_len;
  }
  // It is outside those range, hence the rotation (being positive or negative)
  // won't lead to an out of bounds position.
  else {
    new_pos = rel_pos;
  }

  assert(new_pos < poly_len);
  return new_pos;
}

DEVICE void evaluate_at_pos(GLOBAL FIELD* polys,
                            uint32_t poly_len,
                            GLOBAL Instruction* instructions,
                            uint32_t num_instructions,
                            uint32_t stack_size,
                            GLOBAL FIELD* omega,
                            uint32_t pos,
                            GLOBAL FIELD* result) {
  Stack stack = stack_new(stack_size);

  for (int i = 0; i < num_instructions; i++) {
    Instruction instruction = instructions[i];
    switch (instruction.type) {
      case ADD: {
        const FIELD lhs = stack_pop(&stack);
        const FIELD rhs = stack_pop(&stack);
        const FIELD added = FIELD_add(lhs, rhs);
        stack_push(&stack, added);
        break;
      }
      case MUL: {
        const FIELD lhs = stack_pop(&stack);
        const FIELD rhs = stack_pop(&stack);
        const FIELD multiplied = FIELD_mul(lhs, rhs);
        stack_push(&stack, multiplied);
        break;
      }
      case SCALE: {
        const FIELD lhs = stack_pop(&stack);
        const FIELD scaled = FIELD_mul(lhs, instruction.element);
        stack_push(&stack, scaled);
        break;
      }
      case PUSH: {
        stack_push(&stack, instruction.element);
        break;
      }
      case LINEAR_TERM: {
        const FIELD omega_pow = FIELD_pow(*omega, pos);
        const FIELD linear_term = FIELD_mul(omega_pow, instruction.element);
        stack_push(&stack, linear_term);
        break;
      }
      case ROTATED: {
        uint32_t rotated_pos =
            get_rotated_pos(pos, instruction.rotation, poly_len);
        //  `polys` is a two-dimensional array, but we cannot use it as
        //  usual two-dimensional array as we don't know it's size at
        //  compile time, hence do some pointer arithmetic fun.
        FIELD* rotated = &polys[(instruction.index * poly_len) + rotated_pos];
        stack_push(&stack, *rotated);
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
// KERNEL void evaluate(GLOBAL FIELD polys[][POLY_LEN], GLOBAL FIELD* result,
// uint32_t poly_len) { KERNEL void evaluate(GLOBAL FIELD polys[][], GLOBAL
// Instruction[] instructions, GLOBAL FIELD* omega, uint32_t poly_len, GLOBAL
// FIELD* result) {
KERNEL void evaluate(GLOBAL FIELD* polys,
                     uint32_t poly_len,
                     GLOBAL Instruction* instructions,
                     uint32_t num_instructions,
                     uint32_t stack_size,
                     GLOBAL FIELD* omega,
                     GLOBAL FIELD* result) {
  const uint32_t index = GET_GLOBAL_ID();

  // TODO vmx 2022-10-22: Add the stride to common.cl in ec-gpu-gen and add an
  // OpenCL version.
  const uint stride = blockDim.x * gridDim.x;

  for (int pos = index; pos < poly_len; pos += stride) {
    //  TODO vmx 2022-11-11: check if this if statement is really needed.
    if (pos <= poly_len) {
      evaluate_at_pos(polys, poly_len, instructions, num_instructions,
                      stack_size, omega, pos, &result[pos]);
    }
  }
}
