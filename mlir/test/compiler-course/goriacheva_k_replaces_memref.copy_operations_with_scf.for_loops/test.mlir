// RUN: mlir-opt \
// RUN: -load-pass-plugin=%mlir_lib_dir/goriacheva_k_replaces_memref.copy_operations_with_scf.for_loops_MLIR%shlibext \
// RUN: --pass-pipeline="builtin.module(replace-memref-copy-with-loops)" %s | FileCheck %s

// =======================
// 1D STATIC
// =======================

func.func @copy_1d(%src: memref<4xi32>, %dst: memref<4xi32>) {
  memref.copy %src, %dst : memref<4xi32> to memref<4xi32>
  return
}

// CHECK-LABEL: func.func @copy_1d

// constants
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index

// loop bounds
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]]

// body
// CHECK: memref.load
// CHECK: memref.store

// CHECK-NOT: memref.copy


// =======================
// 2D STATIC (nested loops)
// =======================

func.func @copy_2d(%src: memref<2x3xi32>, %dst: memref<2x3xi32>) {
  memref.copy %src, %dst : memref<2x3xi32> to memref<2x3xi32>
  return
}

// CHECK-LABEL: func.func @copy_2d

// constants
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index

// outer loop
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]]

// inner loop
// CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[C3]] step %[[C1]]

// body
// CHECK: memref.load
// CHECK: memref.store

// CHECK-NOT: memref.copy


// =======================
// DYNAMIC SHAPE
// =======================

func.func @copy_dynamic(%src: memref<?x4xi32>, %dst: memref<?x4xi32>) {
  memref.copy %src, %dst : memref<?x4xi32> to memref<?x4xi32>
  return
}

// CHECK-LABEL: func.func @copy_dynamic

// constants
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index

// dynamic upper bound
// CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]]

// loop
// CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[DIM]] step %[[C1]]

// body
// CHECK: memref.load
// CHECK: memref.store

// CHECK-NOT: memref.copy


// =======================
// MULTIPLE COPIES
// =======================

func.func @multiple_copies(
    %src: memref<4xi32>,
    %tmp: memref<4xi32>,
    %dst: memref<4xi32>) {

  memref.copy %src, %tmp : memref<4xi32> to memref<4xi32>
  memref.copy %tmp, %dst : memref<4xi32> to memref<4xi32>

  return
}

// CHECK-LABEL: func.func @multiple_copies

// first loop
// CHECK: scf.for

// second loop
// CHECK: scf.for

// body ops
// CHECK: memref.load
// CHECK: memref.store

// CHECK-NOT: memref.copy


// =======================
// NO COPY (should stay unchanged)
// =======================

func.func @no_copy(%arg0: memref<4xi32>) {
  % c0 = arith.constant 0 : index % val =
             memref.load % arg0[% c0] : memref<4xi32> memref.store % val,
    % arg0[% c0] : memref<4xi32> return
}

// CHECK-LABEL: func.func @no_copy
// CHECK-NOT: scf.for
// CHECK: memref.load
// CHECK: memref.store
