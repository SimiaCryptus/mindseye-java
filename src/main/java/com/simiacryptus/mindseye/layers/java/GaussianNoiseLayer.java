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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class GaussianNoiseLayer extends LayerBase {

  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(GaussianNoiseLayer.class);
  private long seed = GaussianNoiseLayer.random.get().nextLong();
  private double value;

  public GaussianNoiseLayer() {
    super();
    setValue(1.0);
  }

  protected GaussianNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
  }

  public double getValue() {
    return value;
  }

  @Nonnull
  public GaussianNoiseLayer setValue(final double value) {
    this.value = value;
    return this;
  }

  @SuppressWarnings("unused")
  public static GaussianNoiseLayer fromJson(@Nonnull final JsonObject json,
                                            com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new GaussianNoiseLayer(json);
  }

  public static @SuppressWarnings("unused")
  GaussianNoiseLayer[] addRefs(GaussianNoiseLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(GaussianNoiseLayer::addRef)
        .toArray((x) -> new GaussianNoiseLayer[x]);
  }

  public static @SuppressWarnings("unused")
  GaussianNoiseLayer[][] addRefs(GaussianNoiseLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(GaussianNoiseLayer::addRefs)
        .toArray((x) -> new GaussianNoiseLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0];
    final TensorList inputData = in0.getData();
    final int itemCnt = inputData.length();
    final Tensor[] outputA = com.simiacryptus.ref.wrappers.RefIntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nonnull final Random random = new Random(seed);
      @Nullable final Tensor input = inputData.get(dataIndex);
      return input.map(x -> {
        return x + random.nextGaussian() * getValue();
      });
    }).toArray(i -> new Tensor[i]);
    int[] dimensions = inputData.getDimensions();
    return new Result(new TensorArray(outputA),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (in0.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                  Tensor tensor = delta.get(dataIndex);
                  @Nullable final double[] deltaData = tensor.getData();
                  @Nonnull final Tensor passback = new Tensor(dimensions);
                  for (int i = 0; i < passback.length(); i++) {
                    passback.set(i, deltaData[i]);
                  }
                  return passback;
                }).toArray(i -> new Tensor[i]));
            in0.accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return in0.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }

  public void shuffle() {
    seed = GaussianNoiseLayer.random.get().nextLong();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  GaussianNoiseLayer addRef() {
    return (GaussianNoiseLayer) super.addRef();
  }

}
