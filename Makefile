# --- Toolchain -------------------------------------------------------------
NVCC     ?= nvcc
MPICXX   ?= mpicxx
SM_ARCH  ?= sm_80                  # override on cluster: sm_89 (L40S), sm_90 (H100/H200)
STD       = -std=c++17

NVCC_FLAGS = -O3 -arch=$(SM_ARCH) --use_fast_math \
             --generate-line-info -Xcompiler -Wall,-fPIC $(STD)

INC = -Isrc/common

BUILD = build

# --- Object files ----------------------------------------------------------
KERNEL_OBJS = $(BUILD)/dense.o \
              $(BUILD)/dense_fp16.o \
              $(BUILD)/windowed.o \
              $(BUILD)/windowed_fp16.o \
              $(BUILD)/blocksparse.o \
              $(BUILD)/blocksparse_fp16.o
DISPATCH_OBJ = $(BUILD)/dispatch.o
SPARSE_OBJ   = $(BUILD)/sparse_formats.o
LIB_OBJS    = $(KERNEL_OBJS) $(DISPATCH_OBJ) $(SPARSE_OBJ)

CLI_OBJ      = $(BUILD)/attn_main.o
TEST_OBJS_DENSE       = $(BUILD)/test_dense.o
TEST_OBJS_WINDOWED    = $(BUILD)/test_windowed.o
TEST_OBJS_BLOCKSPARSE = $(BUILD)/test_blocksparse.o

CLI_BIN              = $(BUILD)/attn
TEST_BIN             = $(BUILD)/test_dense
TEST_BIN_WINDOWED    = $(BUILD)/test_windowed
TEST_BIN_BLOCKSPARSE = $(BUILD)/test_blocksparse
MPI_HEADPAR_BIN  = $(BUILD)/attn_mpi_headpar
MPI_SEQPAR_BIN   = $(BUILD)/attn_mpi_seqpar

# MPI compilation flags. nvcc uses mpicxx as the host compiler so that
# MPI headers and libraries are found via the standard MPI compiler wrapper.
# Requires CUDA-aware MPI for the device-pointer Scatter/Gather calls
# (confirmed available in nvhpc/24.1's bundled HPC-X).
MPI_NVCC_FLAGS = $(NVCC_FLAGS) -ccbin $(MPICXX)

# --- Pattern rules ---------------------------------------------------------
$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/kernels/%.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/dispatch.o: src/common/dispatch.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/sparse_formats.o: src/common/sparse_formats.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/attn_main.o: src/cli/attn_main.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/test_dense.o: tests/test_dense.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/test_windowed.o: tests/test_windowed.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/test_blocksparse.o: tests/test_blocksparse.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/head_parallel.o: src/mpi/head_parallel.cu | $(BUILD)
	$(NVCC) $(MPI_NVCC_FLAGS) $(INC) -c -o $@ $<

$(BUILD)/seq_parallel.o: src/mpi/seq_parallel.cu | $(BUILD)
	$(NVCC) $(MPI_NVCC_FLAGS) $(INC) -c -o $@ $<

$(CLI_BIN): $(CLI_OBJ) $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -o $@ $^

$(TEST_BIN): $(TEST_OBJS_DENSE) $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -o $@ $^

$(TEST_BIN_WINDOWED): $(TEST_OBJS_WINDOWED) $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -o $@ $^

$(TEST_BIN_BLOCKSPARSE): $(TEST_OBJS_BLOCKSPARSE) $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -o $@ $^

# MPI binary is linked with mpicxx as the host compiler so libmpi gets pulled in
# automatically. The kernel objects are plain CUDA — linking them here is fine.
$(MPI_HEADPAR_BIN): $(BUILD)/head_parallel.o $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -ccbin $(MPICXX) -o $@ $^

$(MPI_SEQPAR_BIN): $(BUILD)/seq_parallel.o $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) -ccbin $(MPICXX) -o $@ $^

# Shared library exposing extern "C" attn_forward_mha_c so the kernels can be
# called from Python via ctypes (see bench/end_to_end.py). All upstream object
# files are now compiled with -fPIC (see NVCC_FLAGS), so we can link them
# straight into the .so.
LIB_ATTN_SO = $(BUILD)/libattn.so
ATTN_CAPI_OBJ = $(BUILD)/attn_capi.o

$(ATTN_CAPI_OBJ): src/cli/attn_capi.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $(INC) -c -o $@ $<

$(LIB_ATTN_SO): $(ATTN_CAPI_OBJ) $(LIB_OBJS) | $(BUILD)
	$(NVCC) -arch=$(SM_ARCH) --shared -Xcompiler -fPIC -o $@ $^

# --- Phony targets ---------------------------------------------------------
.PHONY: all dense windowed blocksparse mpi mpi-headpar mpi-seqpar attn-lib test test-dense test-windowed test-blocksparse \
        bench profile-ncu profile-nsys validate validate-dense validate-windowed validate-blocksparse clean

all: $(CLI_BIN) $(TEST_BIN) $(TEST_BIN_WINDOWED) $(TEST_BIN_BLOCKSPARSE)

dense:        $(CLI_BIN)
windowed:     $(CLI_BIN) $(TEST_BIN_WINDOWED)
blocksparse:  $(CLI_BIN) $(TEST_BIN_BLOCKSPARSE)
mpi:          mpi-headpar mpi-seqpar
mpi-headpar:  $(MPI_HEADPAR_BIN)
mpi-seqpar:   $(MPI_SEQPAR_BIN)
attn-lib:     $(LIB_ATTN_SO)

test: test-dense test-windowed test-blocksparse

test-dense: $(TEST_BIN)
	$(TEST_BIN)

test-windowed: $(TEST_BIN_WINDOWED)
	$(TEST_BIN_WINDOWED)

test-blocksparse: $(TEST_BIN_BLOCKSPARSE)
	$(TEST_BIN_BLOCKSPARSE)

validate: validate-dense validate-windowed validate-blocksparse

validate-dense: $(CLI_BIN)
	python bench/validate.py --kernel dense

validate-windowed: $(CLI_BIN)
	python bench/validate.py --kernel windowed

validate-blocksparse: $(CLI_BIN)
	python bench/validate.py --kernel blocksparse

bench: $(CLI_BIN)
	python bench/sweep.py --kernel dense

profile-ncu: $(CLI_BIN)
	@mkdir -p analysis/profiling/runs
	ncu --set full --import-source yes \
	    -o analysis/profiling/runs/dense_smoke \
	    $(CLI_BIN) --kernel=dense --N=2048 --d=64 --H=1 --warmup=2 --iters=3

profile-nsys: $(CLI_BIN)
	@mkdir -p analysis/profiling/runs
	nsys profile -o analysis/profiling/runs/dense_smoke -t cuda,nvtx \
	    $(CLI_BIN) --kernel=dense --N=2048 --d=64 --H=16 --warmup=2 --iters=3

clean:
	rm -rf $(BUILD) *.nsys-rep *.ncu-rep
