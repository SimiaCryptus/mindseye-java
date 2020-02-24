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
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class CrossDotMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);

  public CrossDotMetaLayer() {
  }

  protected CrossDotMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static CrossDotMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CrossDotMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList indata = input.getData();
    final int itemCnt = indata.length();
    final int dim = Tensor.length(indata.getDimensions());
    @Nonnull final Tensor results = new Tensor(dim, dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (i == j) {
          continue;
        }
        double v = 0;
        for (int k = 0; k < itemCnt; k++) {
          Tensor tensor = indata.get(k);
          @Nullable final double[] kk = tensor.getData();
          tensor.freeRef();
          v += kk[i] * kk[j];
        }
        results.set(new int[]{i, j}, v);
      }
    }
    TensorArray data = new TensorArray(results);
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(indata, itemCnt, dim, input.getAccumulator(), input.isAlive());
    input.freeRef();
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
  CrossDotMetaLayer addRef() {
    return (CrossDotMetaLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList indata;
    private final int itemCnt;
    private final int dim;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(TensorList indata, int itemCnt, int dim, Result.Accumulator accumulator, boolean alive) {
      this.indata = indata;
      this.itemCnt = itemCnt;
      this.dim = dim;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        @Nullable final Tensor deltaTensor = delta.get(0);
        @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        RefArrays.parallelSetAll(RefUtil.addRefs(feedback), i -> new Tensor(dim));

        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            if (i == j) {
              continue;
            }
            final double v = deltaTensor.get(i, j);
            for (int k = 0; k < itemCnt; k++) {
              Tensor tensor = indata.get(k);
              @Nullable final double[] kk = tensor.getData();
              tensor.freeRef();
              feedback[k].add(i, v * kk[j]);
              feedback[k].add(j, v * kk[i]);
            }
          }
        }
        deltaTensor.freeRef();
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefUtil.addRefs(feedback));
        RefUtil.freeRef(feedback);
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      indata.freeRef();
      accumulator.freeRef();
    }
  }
}
