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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class SubBatchLayer extends WrapperLayer {

  protected SubBatchLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  public SubBatchLayer(final Layer inner) {
    super(inner);
    if (null != inner)
      inner.freeRef();
  }

  @Override
  public RefList<Layer> getChildren() {
    return super.getChildren();
  }

  @SuppressWarnings("unused")
  public static SubBatchLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SubBatchLayer(json, rs);
  }

  public static <T extends Layer> SubBatchLayer wrap(T layer) {
    SubBatchLayer temp_10_0006 = new SubBatchLayer(
        RefUtil.addRef(RefUtil.addRef(layer)));
    if (null != layer)
      layer.freeRef();
    return temp_10_0006;
  }

  public static @SuppressWarnings("unused")
  SubBatchLayer[] addRefs(SubBatchLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SubBatchLayer::addRef)
        .toArray((x) -> new SubBatchLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SubBatchLayer[][] addRefs(SubBatchLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SubBatchLayer::addRefs)
        .toArray((x) -> new SubBatchLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(final Result... inputs) {
    Layer inner = getInner();
    TensorList temp_10_0008 = inputs[0].getData();
    int batches = temp_10_0008.length();
    if (null != temp_10_0008)
      temp_10_0008.freeRef();
    Tensor[][] passbackBuffer = RefIntStream.range(0, inputs.length)
        .mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends Tensor[]>) inputIndex -> {
              return new Tensor[inputs[inputIndex].getData().length()];
            }, Result.addRefs(inputs)))
        .toArray(x -> new Tensor[x][]);
    Result[] batchResults = RefIntStream.range(0, batches).mapToObj(RefUtil
        .wrapInterface((IntFunction<? extends Result>) batchIndex -> {
              return inner
                  .eval(RefIntStream.range(0, inputs.length).mapToObj(RefUtil.wrapInterface(
                      (IntFunction<? extends Result>) inputIndex -> {
                        TensorList temp_10_0010 = inputs[inputIndex].getData();
                        Result temp_10_0009 = new Result(
                            new TensorArray(temp_10_0010.get(batchIndex)), new Result.Accumulator() {
                          {
                          }

                          @Override
                          public void accept(DeltaSet<UUID> deltaBuffer, TensorList deltaSignal) {
                            if (null != deltaBuffer)
                              deltaBuffer.freeRef();
                            {
                              Tensor temp_10_0001 = deltaSignal.get(0);
                              if (null != passbackBuffer[inputIndex][batchIndex])
                                passbackBuffer[inputIndex][batchIndex].freeRef();
                              passbackBuffer[inputIndex][batchIndex] = temp_10_0001 == null ? null
                                  : temp_10_0001.addRef();
                              if (null != temp_10_0001)
                                temp_10_0001.freeRef();
                            }
                            if (null != deltaSignal)
                              deltaSignal.freeRef();
                          }

                          public @SuppressWarnings("unused")
                          void _free() {
                          }
                        });
                        if (null != temp_10_0010)
                          temp_10_0010.freeRef();
                        return temp_10_0009;
                      }, Tensor.addRefs(passbackBuffer),
                      Result.addRefs(inputs))).<Result>toArray(x -> new Result[x]));
            }, Tensor.addRefs(passbackBuffer),
            Result.addRefs(inputs), inner == null ? null : inner.addRef()))
        .toArray(i -> new Result[i]);
    if (null != inner)
      inner.freeRef();
    TensorArray resultData = new TensorArray(
        RefArrays.stream(Result.addRefs(batchResults)).map(x -> {
          TensorList temp_10_0011 = x.getData();
          Tensor temp_10_0004 = temp_10_0011.get(0);
          if (null != temp_10_0011)
            temp_10_0011.freeRef();
          if (null != x)
            x.freeRef();
          return temp_10_0004;
        }).toArray(i -> new Tensor[i]));
    try {
      try {
        try {
          try {
            return new Result(resultData, new Result.Accumulator() {
              {
                Result.addRefs(inputs);
              }

              @Override
              public void accept(DeltaSet<UUID> deltaBuffer, TensorList deltaSignal) {
                RefIntStream.range(0, deltaSignal.length()).forEach(
                    RefUtil.wrapInterface(batchIndex -> {
                          TensorArray tensorArray = new TensorArray(deltaSignal.get(batchIndex));
                          Result.Accumulator temp_10_0012 = batchResults[batchIndex]
                              .getAccumulator();
                          temp_10_0012.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                              tensorArray == null ? null : tensorArray.addRef());
                          if (null != temp_10_0012)
                            temp_10_0012.freeRef();
                          if (null != tensorArray)
                            tensorArray.freeRef();
                        }, deltaSignal == null ? null : deltaSignal.addRef(),
                        Result.addRefs(batchResults),
                        deltaBuffer == null ? null : deltaBuffer.addRef()));
                if (null != deltaSignal)
                  deltaSignal.freeRef();
                synchronized (passbackBuffer) {
                  RefIntStream.range(0, inputs.length).forEach(
                      RefUtil.wrapInterface(inputIndex -> {
                            TensorArray tensorArray = new TensorArray(
                                Tensor.addRefs(passbackBuffer[inputIndex]));
                            Result.Accumulator temp_10_0013 = inputs[inputIndex]
                                .getAccumulator();
                            temp_10_0013.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                                tensorArray == null ? null : tensorArray.addRef());
                            if (null != temp_10_0013)
                              temp_10_0013.freeRef();
                            if (null != tensorArray)
                              tensorArray.freeRef();
                          }, Tensor.addRefs(passbackBuffer),
                          Result.addRefs(inputs),
                          deltaBuffer == null ? null : deltaBuffer.addRef()));
                }
                if (null != deltaBuffer)
                  deltaBuffer.freeRef();
              }

              public @SuppressWarnings("unused")
              void _free() {
                if (null != inputs)
                  ReferenceCounting.freeRefs(inputs);
              }
            }) {
              public void _free() {
                super._free();
              }
            };
          } finally {
            if (null != inputs)
              ReferenceCounting.freeRefs(inputs);
          }
        } finally {
          if (null != resultData)
            resultData.freeRef();
        }
      } finally {
        if (null != batchResults)
          ReferenceCounting.freeRefs(batchResults);
      }
    } finally {
      if (null != passbackBuffer)
        ReferenceCounting.freeRefs(passbackBuffer);
    }
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SubBatchLayer addRef() {
    return (SubBatchLayer) super.addRef();
  }

}
