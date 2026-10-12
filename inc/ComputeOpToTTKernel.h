/**
 * @file ComputeOpToTTKernel.h
 * @brief Declarations for compute op conversion patterns to TTKernel.
 */

#pragma once

#include "FuncOpToTTKernel.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace loom {
class BroadcastOp;
} // namespace loom

namespace mlir::loom {

/// Transport synchronization protocol for cross-core reduce operations.
enum class ReduceProtocol {
  MultiSlot,
  SingleSlot
};

/// @deprecated Use ReduceProtocol instead.
using ReduceSumProtocol = ReduceProtocol;

/**
 * @brief Populate conversion patterns for compute ops (e.g., linalg.matmul).
 *
 * @param patterns Pattern set to populate.
 * @param typeConverter Type converter used for the conversion pipeline.
 * @param context MLIR context.
 */
void populateComputeOpConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::MLIRContext *context,
                                         std::shared_ptr<CompileArgTracker>
                                             tracker);

/// Returns true when this `linalg.generic` matches one of the supported
/// FlashAttention compute forms handled by this pass.
bool isSupportedFlashAttentionGeneric(mlir::linalg::GenericOp op);

/// Folds each loom.broadcast whose result only feeds elementwise generics into
/// those generics: they read the broadcast source through a broadcasting
/// indexing map, add/sub/mul take it as the broadcast operand of an FPU tile
/// op, and a division by it becomes one reciprocal of the source and a
/// multiply. A broadcast with any other use stays materialized.
void foldMaterializedBroadcasts(mlir::ModuleOp module);

/// Erases each zero fill whose buffer is next touched as the output of a
/// matmul. The matmul lowering overwrites its output on the first k block
/// (L1 accumulation starts on the second), so the filled tiles are never read.
void eraseOverwrittenAccumulatorFills(mlir::ModuleOp module);

/// Returns true when this `linalg.copy` is a compute-kernel copy handled by
/// this pass.
bool shouldConvertComputeLinalgCopy(mlir::linalg::CopyOp op);

/// Returns true when this `linalg.transpose` is a compute-kernel transpose
/// handled by this pass.
bool shouldConvertComputeLinalgTranspose(mlir::linalg::TransposeOp op);

} // namespace mlir::loom
