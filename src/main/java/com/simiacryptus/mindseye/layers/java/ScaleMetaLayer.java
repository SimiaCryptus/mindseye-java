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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class ScaleMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);

  public ScaleMetaLayer() {
  }

  protected ScaleMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static ScaleMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleMetaLayer(json);
  }

  public static @SuppressWarnings("unused")
  ScaleMetaLayer[] addRefs(ScaleMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ScaleMetaLayer::addRef)
        .toArray((x) -> new ScaleMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ScaleMetaLayer[][] addRefs(ScaleMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ScaleMetaLayer::addRefs)
        .toArray((x) -> new ScaleMetaLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    final Result in1 = inObj[1].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList data0 = in0.getData();
    final TensorList data1 = in1.getData();
    final int itemCnt = data0.length();
    final Tensor data10 = data1.get(0);
    if (null != data1)
      data1.freeRef();
    final Tensor[] tensors = RefIntStream.range(0, itemCnt).mapToObj(RefUtil
        .wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          return data0.get(dataIndex)
              .mapIndex(RefUtil.wrapInterface(
                  (v, c) -> v * data10.get(c),
                  data10 == null ? null : data10.addRef()));
        }, data10 == null ? null : data10.addRef(), data0 == null ? null : data0.addRef())).toArray(i -> new Tensor[i]);
    if (null != data0)
      data0.freeRef();
    Tensor tensor0 = tensors[0].addRef();
    try {
      try {
        try {
          try {
            try {
              return new Result(new TensorArray(Tensor.addRefs(tensors)),
                  new Result.Accumulator() {
                    {
                    }

                    @Override
                    public void accept(DeltaSet<UUID> buffer, TensorList data) {
                      if (in0.isAlive()) {
                        @Nonnull
                        TensorArray tensorArray = new TensorArray(
                            data.stream().map(RefUtil.wrapInterface(
                                (Function<? super Tensor, ? extends Tensor>) t -> {
                                  Tensor temp_56_0006 = t
                                      .mapIndex(RefUtil.wrapInterface(
                                          (v, c) -> {
                                            return v * data10.get(c);
                                          }, data10 == null ? null : data10.addRef()));
                                  if (null != t)
                                    t.freeRef();
                                  return temp_56_0006;
                                }, data10 == null ? null : data10.addRef())).toArray(i -> new Tensor[i]));
                        in0.accumulate(buffer == null ? null : buffer.addRef(),
                            tensorArray == null ? null : tensorArray);
                      }
                      if (in1.isAlive()) {
                        @Nullable final Tensor passback = tensor0.mapIndex(RefUtil
                            .wrapInterface((v, c) -> {
                              return RefIntStream.range(0, itemCnt).mapToDouble(RefUtil
                                  .wrapInterface(i -> {
                                    return data.get(i).get(c) * data.get(i).get(c);
                                  }, data == null ? null : data.addRef())).sum();
                            }, data == null ? null : data.addRef()));
                        @Nonnull
                        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length())
                            .mapToObj(RefUtil.wrapInterface(
                                (IntFunction<? extends Tensor>) i -> i == 0
                                    ? passback
                                    : passback.map(v -> 0),
                                passback == null ? null : passback.addRef()))
                            .toArray(i -> new Tensor[i]));
                        if (null != passback)
                          passback.freeRef();
                        in1.accumulate(buffer == null ? null : buffer.addRef(),
                            tensorArray == null ? null : tensorArray);
                      }
                      if (null != data)
                        data.freeRef();
                      if (null != buffer)
                        buffer.freeRef();
                    }

                    public @SuppressWarnings("unused")
                    void _free() {
                    }
                  }) {

                {
                }

                @Override
                public boolean isAlive() {
                  return in0.isAlive() || in1.isAlive();
                }

                public void _free() {
                }

              };
            } finally {
              if (null != tensor0)
                tensor0.freeRef();
            }
          } finally {
            if (null != tensors)
              ReferenceCounting.freeRefs(tensors);
          }
        } finally {
          if (null != data10)
            data10.freeRef();
        }
      } finally {
        if (null != in1)
          in1.freeRef();
      }
    } finally {
      if (null != in0)
        in0.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ScaleMetaLayer addRef() {
    return (ScaleMetaLayer) super.addRef();
  }
}
