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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class BiasMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasMetaLayer.class);

  public BiasMetaLayer() {
  }

  protected BiasMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BiasMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    final Result in1 = inObj[1].addRef();
    RefUtil.freeRef(inObj);
    TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    TensorArray data = fwd(in1.addRef(), data0, itemCnt);
    boolean alive = in0.isAlive() || in1.isAlive();
    Result.Accumulator accumulator = new Accumulator(data.get(0), itemCnt, in0.getAccumulator(), in0.isAlive(), in1.getAccumulator(), in1.isAlive());
    in0.freeRef();
    in1.freeRef();
    return new Result(data, accumulator, alive);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BiasMetaLayer addRef() {
    return (BiasMetaLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(Result in1, TensorList data0, int itemCnt) {
    final TensorList data1 = in1.getData();
    in1.freeRef();
    Tensor tensor1 = data1.get(0);
    data1.freeRef();
    return new TensorArray(RefIntStream.range(0, itemCnt).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor tensor = data0.get(dataIndex);
          Tensor temp_48_0003 = tensor.mapIndex(RefUtil.wrapInterface((v, c) -> {
            return v + tensor1.get(c);
          }, tensor1.addRef()));
          tensor.freeRef();
          return temp_48_0003;
        }, tensor1, data0))
        .toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor tensor0;
    private final int itemCnt;
    private Result.Accumulator accumulator1;
    private boolean alive1;
    private Result.Accumulator accumulator0;
    private boolean alive0;

    public Accumulator(Tensor tensor0, int itemCnt, Result.Accumulator accumulator0, boolean alive0, Result.Accumulator accumulator1, boolean alive1) {
      this.tensor0 = tensor0;
      this.itemCnt = itemCnt;
      this.accumulator1 = accumulator1;
      this.alive1 = alive1;
      this.accumulator0 = accumulator0;
      this.alive0 = alive0;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive1) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
              if (i == 0)
                return tensor0.mapCoords(RefUtil.wrapInterface(c -> {
                  return RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(j -> {
                    Tensor tensor = data.get(j);
                    double temp_48_0006 = tensor.get(c);
                    tensor.freeRef();
                    return temp_48_0006;
                  }, data.addRef())).sum();
                }, data.addRef()));
              else {
                return tensor0.mapCoords(v -> 0);
              }
            }, data.addRef(), tensor0.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator1.accept(buffer1, tensorArray);
      }
      if (alive0) {
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        accumulator0.accept(buffer1, data.addRef());
      }
      data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator0.freeRef();
      accumulator1.freeRef();
      tensor0.freeRef();
    }
  }
}
