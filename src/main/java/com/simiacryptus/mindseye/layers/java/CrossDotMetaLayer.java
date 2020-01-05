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
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class CrossDotMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);

  public CrossDotMetaLayer() {
  }

  protected CrossDotMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static CrossDotMetaLayer fromJson(@Nonnull final JsonObject json,
                                           Map<CharSequence, byte[]> rs) {
    return new CrossDotMetaLayer(json);
  }

  public static @SuppressWarnings("unused")
  CrossDotMetaLayer[] addRefs(CrossDotMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CrossDotMetaLayer::addRef)
        .toArray((x) -> new CrossDotMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  CrossDotMetaLayer[][] addRefs(CrossDotMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(CrossDotMetaLayer::addRefs)
        .toArray((x) -> new CrossDotMetaLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
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
          v += kk[i] * kk[j];
        }
        results.set(new int[]{i, j}, v);
      }
    }
    return new Result(new TensorArray(results),
        new Result.Accumulator() {
          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList delta) {
            if (input.isAlive()) {
              @Nullable final Tensor deltaTensor = delta.get(0);
              @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
              RefArrays.parallelSetAll(feedback, i -> new Tensor(dim));

              for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                  if (i == j) {
                    continue;
                  }
                  final double v = deltaTensor.get(i, j);
                  for (int k = 0; k < itemCnt; k++) {
                    Tensor tensor = indata.get(k);
                    @Nullable final double[] kk = tensor.getData();
                    feedback[k].add(i, v * kk[j]);
                    feedback[k].add(j, v * kk[i]);
                  }
                }
              }
              @Nonnull
              TensorArray tensorArray = new TensorArray(feedback);
              input.accumulate(buffer, tensorArray);
            }
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
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
  CrossDotMetaLayer addRef() {
    return (CrossDotMetaLayer) super.addRef();
  }
}
