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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.WrapperLayer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class SubBatchLayer extends WrapperLayer {

  protected SubBatchLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  public SubBatchLayer(final Layer inner) {
    super(inner);
  }

  public static SubBatchLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SubBatchLayer(json, rs);
  }

  public static <T extends Layer> SubBatchLayer wrap(T layer) {
    SubBatchLayer subBatchLayer = new SubBatchLayer(layer);
    layer.freeRef();
    return subBatchLayer;
  }

  @Override
  public List<Layer> getChildren() {
    return super.getChildren();
  }

  @Nullable
  @Override
  public Result eval(final Result... inputs) {
    Layer inner = getInner();
    inner.addRef();
    Arrays.stream(inputs).forEach(ReferenceCountingBase::addRef);
    int batches = inputs[0].getData().length();
    Tensor[][] passbackBuffer = IntStream.range(0, inputs.length)
        .mapToObj(inputIndex -> new Tensor[inputs[inputIndex].getData().length()])
        .toArray(x -> new Tensor[x][]);
    Result[] batchResults = IntStream.range(0, batches).mapToObj(batchIndex -> {
      return inner.evalAndFree(IntStream.range(0, inputs.length).mapToObj(inputIndex -> {
        return new Result(
            TensorArray.wrap(inputs[inputIndex].getData().get(batchIndex)),
            (deltaBuffer, deltaSignal) -> {
              passbackBuffer[inputIndex][batchIndex] = deltaSignal.get(0);
              deltaSignal.freeRef();
            });
      }).toArray(x -> new Result[x]));
    }).toArray(i -> new Result[i]);
    TensorArray resultData = TensorArray.wrap(Arrays.stream(batchResults).map(x -> x.getData().get(0)).toArray(i -> new Tensor[i]));
    Arrays.stream(batchResults).map(Result::getData).forEach(ReferenceCounting::freeRef);
    return new Result(
        resultData,
        (DeltaSet<UUID> deltaBuffer, TensorList deltaSignal) -> {
          try {
            IntStream.range(0, deltaSignal.length()).forEach(batchIndex -> {
              TensorArray tensorArray = TensorArray.wrap(deltaSignal.get(batchIndex));
              batchResults[batchIndex].getAccumulator().accept(deltaBuffer, tensorArray);
            });
          } finally {
            deltaSignal.freeRef();
          }
          synchronized (passbackBuffer) {
            IntStream.range(0, inputs.length).forEach(inputIndex -> {
              TensorArray tensorArray = TensorArray.wrap(passbackBuffer[inputIndex]);
              inputs[inputIndex].getAccumulator().accept(deltaBuffer, tensorArray);
            });
          }
        }) {
      @Override
      protected void _free() {
        Arrays.stream(inputs).forEach(ReferenceCounting::freeRef);
        Arrays.stream(batchResults).forEach(ReferenceCounting::freeRef);
        inner.freeRef();
        super._free();
      }
    };
  }

  @Nullable
  @Override
  public Result evalAndFree(final Result... inputs) {
    Layer inner = getInner();
    int batches = inputs[0].getData().length();
    Tensor[][] passbackBuffer = IntStream.range(0, inputs.length)
        .mapToObj(inputIndex -> new Tensor[inputs[inputIndex].getData().length()])
        .toArray(x -> new Tensor[x][]);
    Result[] batchResults = IntStream.range(0, batches).mapToObj(batchIndex -> {
      return inner.evalAndFree(IntStream.range(0, inputs.length).mapToObj(inputIndex -> {
        return new Result(
            TensorArray.wrap(inputs[inputIndex].getData().get(batchIndex)),
            (deltaBuffer, deltaSignal) -> {
              passbackBuffer[inputIndex][batchIndex] = deltaSignal.get(0);
            });
      }).toArray(x -> new Result[x]));
    }).toArray(i -> new Result[i]);
    Arrays.stream(inputs).map(Result::getData).forEach(ReferenceCounting::freeRef);
    TensorArray resultData = TensorArray.wrap(Arrays.stream(batchResults).map(x -> x.getData().get(0)).toArray(i -> new Tensor[i]));
    Arrays.stream(batchResults).map(Result::getData).forEach(ReferenceCounting::freeRef);
    return new Result(
        resultData,
        (deltaBuffer, deltaSignal) -> {
          synchronized (passbackBuffer) {
            IntStream.range(0, deltaSignal.length()).forEach(batchIndex -> {
              TensorArray tensorArray = TensorArray.wrap(deltaSignal.get(batchIndex));
              batchResults[batchIndex].getAccumulator().accept(deltaBuffer, tensorArray);
              tensorArray.freeRef();
            });
            IntStream.range(0, inputs.length).forEach(inputIndex -> {
              TensorArray tensorArray = TensorArray.wrap(passbackBuffer[inputIndex]);
              inputs[inputIndex].getAccumulator().accept(deltaBuffer, tensorArray);
              tensorArray.freeRef();
            });
          }
        }) {
      @Override
      protected void _free() {
        Arrays.stream(inputs).forEach(ReferenceCounting::freeRef);
        Arrays.stream(batchResults).forEach(ReferenceCounting::freeRef);
        super._free();
      }
    };
  }


}
