# Makefile for Benchmark_TransformFunctions1

# Toolchain definitions
RISCV_GCC     := gcc
RISCV_OBJDUMP := objdump
RISCV_OBJCOPY := objcopy

# Directories
SRC_DIR       := .
INC_DIR       := ../../inc
BUILD_DIR     := ./build
TRANSFORM_DIR := ../../src/TransformFunctions
COMMON_DIR    := ../../src/CommonTables

# Source files
SRC := $(SRC_DIR)/Benchmark_TransformFunctions1.c
SRC += $(wildcard $(TRANSFORM_DIR)/*.c)
SRC += $(wildcard $(COMMON_DIR)/*.c)

# Object files
OBJ := $(patsubst %.c,$(BUILD_DIR)/%.o,$(notdir $(SRC)))

# Binary output
TARGET := $(BUILD_DIR)/Benchmark_TransformFunctions1.elf

# Compiler flags
CFLAGS := -O3 -g -w # -march=rv32imac -mabi=ilp32 -O3 -g
CFLAGS += -I$(INC_DIR) -I$(TRANSFORM_DIR) -I$(COMMON_DIR)
CFLAGS += -DPRINT_OUTPUT  # Comment this line to disable output printing

# Linker flags
LDFLAGS := #-T ../../link.ld

# Default target
all: $(TARGET)

# Rule to create build directory
$(BUILD_DIR):
	mkdir -p $@

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(RISCV_GCC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TRANSFORM_DIR)/%.c | $(BUILD_DIR)
	$(RISCV_GCC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(COMMON_DIR)/%.c | $(BUILD_DIR)
	$(RISCV_GCC) $(CFLAGS) -c $< -o $@

# Link object files
$(TARGET): $(OBJ)
	$(RISCV_GCC) $(CFLAGS) $(LDFLAGS) $^ -o $@
	$(RISCV_OBJDUMP) -D $@ > $(BUILD_DIR)/Benchmark_TransformFunctions1.dump
	$(RISCV_OBJCOPY) -O binary $@ $(BUILD_DIR)/Benchmark_TransformFunctions1.bin

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
