/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class SubBatchLayer extends WrapperLayer {

  protected SubBatchLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  public SubBatchLayer(final Layer inner) {
    super(inner);
  }

  @Override
  public RefList<Layer> getChildren() {
    return super.getChildren();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SubBatchLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SubBatchLayer(json, rs);
  }

  @Nonnull
  public static <T extends Layer> SubBatchLayer wrap(@Nullable T layer) {
    return new SubBatchLayer(layer);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inputs) {
    Layer inner = getInner();
    TensorList data0 = inputs[0].getData();
    int batches = data0.length();
    data0.freeRef();
    Tensor[][] passbackBuffer = RefIntStream.range(0, inputs.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor[]>) inputIndex -> {
          TensorList data = inputs[inputIndex].getData();
          Tensor[] tensors = new Tensor[data.length()];
          data.freeRef();
          return tensors;
        }, RefUtil.addRefs(inputs))).toArray(Tensor[][]::new);
    Result[] batchResults = RefIntStream.range(0, batches)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Result>) batchIndex -> {
          assert inner != null;
          return inner.eval(RefIntStream.range(0, inputs.length)
              .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Result>) inputIndex -> {
                Tensor[] tensors = RefUtil.addRefs(passbackBuffer[inputIndex]);
                Result.Accumulator accumulator = new SubAccumulator(tensors, batchIndex);
                TensorList tensorList = inputs[inputIndex].getData();
                Result result = new Result(new TensorArray(tensorList.get(batchIndex)), accumulator);
                tensorList.freeRef();
                return result;
              }, RefUtil.addRefs(passbackBuffer), RefUtil.addRefs(inputs))).<Result>toArray(Result[]::new));
        }, RefUtil.addRefs(passbackBuffer), RefUtil.addRefs(inputs), inner == null ? null : inner.addRef()))
        .toArray(Result[]::new);
    if (null != inner)
      inner.freeRef();
    TensorArray resultData = new TensorArray(RefArrays.stream(RefUtil.addRefs(batchResults)).map(result -> {
      TensorList tensorList = result.getData();
      result.freeRef();
      Tensor tensor = tensorList.get(0);
      tensorList.freeRef();
      return tensor;
    }).toArray(Tensor[]::new));
    return new Result(resultData, new MainAccumulator(batchResults, passbackBuffer, inputs));
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SubBatchLayer addRef() {
    return (SubBatchLayer) super.addRef();
  }

  private static class SubAccumulator extends Result.Accumulator {

    private final Tensor[] tensors;
    private final int batchIndex;

    public SubAccumulator(Tensor[] tensors, int batchIndex) {
      this.tensors = tensors;
      this.batchIndex = batchIndex;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nonnull TensorList deltaSignal) {
      if (null != deltaBuffer)
        deltaBuffer.freeRef();
      RefUtil.set(tensors, batchIndex, deltaSignal.get(0));
      deltaSignal.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(tensors);
    }
  }

  private static class MainAccumulator extends Result.Accumulator {

    private final Result[] batchResults;
    private final Tensor[][] passbackBuffer;
    private final Result[] inputs;

    public MainAccumulator(Result[] batchResults, Tensor[][] passbackBuffer, Result... inputs) {
      this.batchResults = batchResults;
      this.passbackBuffer = passbackBuffer;
      this.inputs = inputs;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nonnull TensorList deltaSignal) {
      RefIntStream.range(0, deltaSignal.length()).forEach(RefUtil.wrapInterface(batchIndex -> {
            TensorArray tensorArray = new TensorArray(deltaSignal.get(batchIndex));
            Result.Accumulator temp_10_0012 = batchResults[batchIndex].getAccumulator();
            assert temp_10_0012 != null;
            temp_10_0012.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                tensorArray.addRef());
            temp_10_0012.freeRef();
            tensorArray.freeRef();
          }, deltaSignal.addRef(), RefUtil.addRefs(batchResults),
          deltaBuffer == null ? null : deltaBuffer.addRef()));
      deltaSignal.freeRef();
      synchronized (passbackBuffer) {
        RefIntStream.range(0, inputs.length).forEach(RefUtil.wrapInterface(inputIndex -> {
              TensorArray tensorArray = new TensorArray(RefUtil.addRefs(passbackBuffer[inputIndex]));
              Result.Accumulator temp_10_0013 = inputs[inputIndex].getAccumulator();
              assert temp_10_0013 != null;
              temp_10_0013.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                  tensorArray.addRef());
              temp_10_0013.freeRef();
              tensorArray.freeRef();
            }, RefUtil.addRefs(passbackBuffer), RefUtil.addRefs(inputs),
            deltaBuffer == null ? null : deltaBuffer.addRef()));
      }
      if (null != deltaBuffer)
        deltaBuffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inputs);
      RefUtil.freeRef(passbackBuffer);
      RefUtil.freeRef(batchResults);
    }
  }
}
