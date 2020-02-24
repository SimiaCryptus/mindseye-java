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
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ScaleMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);

  public ScaleMetaLayer() {
  }

  protected ScaleMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ScaleMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    final Result in1 = inObj[1].addRef();
    RefUtil.freeRef(inObj);
    final TensorList data0 = in0.getData();
    final TensorList data1 = in1.getData();
    final int itemCnt = data0.length();
    final Tensor scale = data1.get(0);
    data1.freeRef();
    int[] dimensions = data0.getDimensions();
    TensorArray data = fwd(data0, itemCnt, scale.addRef());
    boolean alive = in0.isAlive() || in1.isAlive();
    Result.Accumulator accumulator = new Accumulator(scale, dimensions, itemCnt, in0.getAccumulator(), in0.isAlive(), in1.isAlive(), in1.getAccumulator());
    in0.freeRef();
    in1.freeRef();
    return new Result(data, accumulator, alive);
  }

  @NotNull
  private TensorArray fwd(TensorList data0, int itemCnt, Tensor scale) {
    return new TensorArray(RefIntStream.range(0, itemCnt)
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            Tensor tensor = data0.get(dataIndex);
            Tensor tensor1 = tensor.mapIndex((v, c) -> v * scale.get(c));
            tensor.freeRef();
            return tensor1;
          }, scale, data0)).toArray(Tensor[]::new));
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
  ScaleMetaLayer addRef() {
    return (ScaleMetaLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor scale;
    private final int[] dimensions;
    private final int itemCnt;
    private Result.Accumulator accumulator0;
    private boolean alive0;
    private boolean alive1;
    private Result.Accumulator accumulator1;

    public Accumulator(Tensor scale, int[] dimensions, int itemCnt, Result.Accumulator accumulator0, boolean alive0, boolean alive1, Result.Accumulator accumulator1) {
      this.scale = scale;
      this.dimensions = dimensions;
      this.itemCnt = itemCnt;
      this.accumulator0 = accumulator0;
      this.alive0 = alive0;
      this.alive1 = alive1;
      this.accumulator1 = accumulator1;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive0) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            data.stream().map(RefUtil.wrapInterface((Function<? super Tensor, ? extends Tensor>) t -> {
              Tensor temp_56_0006 = t.mapIndex(RefUtil.wrapInterface((v, c) -> {
                return v * scale.get(c);
              }, scale.addRef()));
              t.freeRef();
              return temp_56_0006;
            }, scale.addRef())).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          this.accumulator0.accept(buffer1, tensorArray);
      }
      if (alive1) {
        int length = data.length();
        Tensor indices = new Tensor(dimensions);
        @Nullable final Tensor passback = indices.mapCoords(RefUtil.wrapInterface((c) -> {
          return RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(i -> {
            Tensor tensor = data.get(i);
            double v1 = tensor.get(c) * tensor.get(c);
            tensor.freeRef();
            return v1;
          }, data.addRef())).sum();
        }, data));
        indices.freeRef();
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, length)
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
              return i == 0 ? passback.addRef() : passback.map(v -> 0);
            }, passback)).toArray(Tensor[]::new));
          accumulator1.accept(buffer, tensorArray);
      } else {
        data.freeRef();
        if (null != buffer)
          buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      scale.freeRef();
      accumulator0.freeRef();
      accumulator1.freeRef();
    }
  }
}
