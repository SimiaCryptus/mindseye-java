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
    TensorList temp_10_0008 = inputs[0].getData();
    int batches = temp_10_0008.length();
    temp_10_0008.freeRef();
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
                Result.Accumulator accumulator = new Result.Accumulator() {
                  {
                    RefUtil.addRefs(tensors);
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
                };
                RefUtil.freeRef(tensors);
                TensorList temp_10_0010 = inputs[inputIndex].getData();
                Result temp_10_0009 = new Result(new TensorArray(temp_10_0010.get(batchIndex)), accumulator);
                temp_10_0010.freeRef();
                return temp_10_0009;
              }, RefUtil.addRefs(passbackBuffer), RefUtil.addRefs(inputs))).<Result>toArray(Result[]::new));
        }, RefUtil.addRefs(passbackBuffer), RefUtil.addRefs(inputs), inner == null ? null : inner.addRef()))
        .toArray(Result[]::new);
    if (null != inner)
      inner.freeRef();
    TensorArray resultData = new TensorArray(RefArrays.stream(RefUtil.addRefs(batchResults)).map(x -> {
      TensorList temp_10_0011 = x.getData();
      Tensor temp_10_0004 = temp_10_0011.get(0);
      temp_10_0011.freeRef();
      x.freeRef();
      return temp_10_0004;
    }).toArray(Tensor[]::new));
    try {
      return new Result(resultData, new Result.Accumulator() {
        {
          RefUtil.addRefs(inputs);
          RefUtil.addRefs(batchResults);
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
      });
    } finally {
      RefUtil.freeRef(inputs);
      RefUtil.freeRef(batchResults);
      RefUtil.freeRef(passbackBuffer);
    }
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

}
