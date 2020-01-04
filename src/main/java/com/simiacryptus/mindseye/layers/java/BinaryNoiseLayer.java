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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Random;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class BinaryNoiseLayer extends LayerBase implements StochasticComponent {

  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BinaryNoiseLayer.class);
  @Nonnull
  final com.simiacryptus.ref.wrappers.RefList<Tensor> maskList = new com.simiacryptus.ref.wrappers.RefArrayList<>();
  private double value;
  private long seed = System.nanoTime();

  public BinaryNoiseLayer() {
    this(0.5);
  }

  public BinaryNoiseLayer(final double value) {
    super();
    setValue(value);
  }

  protected BinaryNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
    seed = json.get("seed").getAsLong();
    JsonElement enabled = json.get("enabled");
    //    this.enabled = enabled == null || enabled.getAsBoolean();
  }

  public double getValue() {
    return value;
  }

  @Nonnull
  public void setValue(final double value) {
    this.value = value;
    shuffle(StochasticComponent.random.get().nextLong());
  }

  @SuppressWarnings("unused")
  public static BinaryNoiseLayer fromJson(@Nonnull final JsonObject json,
                                          com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new BinaryNoiseLayer(json);
  }

  @NotNull
  public static Layer maskLayer(double density) {
    PipelineNetwork subnet = new PipelineNetwork(1);
    subnet.add(new ProductInputsLayer(), subnet.add(new BinaryNoiseLayer(density), subnet.getInput(0)),
        subnet.getInput(0));
    return subnet;
  }

  public static @SuppressWarnings("unused")
  BinaryNoiseLayer[] addRefs(BinaryNoiseLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BinaryNoiseLayer::addRef)
        .toArray((x) -> new BinaryNoiseLayer[x]);
  }

  public static @SuppressWarnings("unused")
  BinaryNoiseLayer[][] addRefs(BinaryNoiseLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BinaryNoiseLayer::addRefs)
        .toArray((x) -> new BinaryNoiseLayer[x][]);
  }

  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
    TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (maskList.size() > 0
        && !com.simiacryptus.ref.wrappers.RefArrays.equals(maskList.get(0).getDimensions(), dimensions)) {
      clear();
    }
    @Nonnull final Tensor tensorPrototype = new Tensor(dimensions);
    double amplitude = 1.0 / getValue();
    while (length > maskList.size()) {
      if (seed == 0) {
        maskList.add(tensorPrototype.map(v -> amplitude, false));
      } else {
        Random random = new Random(seed * maskList.size());
        maskList.add(tensorPrototype.map(v -> random.nextDouble() < getValue() ? amplitude : 0, false));
      }
    }
    TensorArray data = new TensorArray(maskList.stream().limit(length).toArray(i -> new Tensor[i]));
    assert inputData.length() == data.length() : (inputData.length() + " != " + data.length());
    return new Result(data, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      input.accumulate(buffer, new TensorArray(delta.stream().map(t -> t.map(x -> 0)).toArray(i -> new Tensor[i])));
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      public void _free() {
      }
    };
  }

  public void clear() {
    final com.simiacryptus.ref.wrappers.RefList<Tensor> maskList = this.maskList;
    synchronized (maskList) {
      maskList.clear();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    json.addProperty("seed", seed);
    //    json.addProperty("enabled", enabled);
    return json;
  }

  @Override
  public void shuffle(final long seed) {
    clear();
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    clear();
    this.seed = 0;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public void _free() {
    clear();
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  BinaryNoiseLayer addRef() {
    return (BinaryNoiseLayer) super.addRef();
  }

}
