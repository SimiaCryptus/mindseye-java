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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * The type Binary noise layer.
 */
@SuppressWarnings("serial")
public class BinaryNoiseLayer extends LayerBase implements StochasticComponent {

  /**
   * The constant random.
   */
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Nonnull
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BinaryNoiseLayer.class);
  /**
   * The Mask list.
   */
  @Nonnull
  final RefList<Tensor> maskList = new RefArrayList<>();
  private double value;
  private long seed = RefSystem.nanoTime();

  /**
   * Instantiates a new Binary noise layer.
   */
  public BinaryNoiseLayer() {
    this(0.5);
  }

  /**
   * Instantiates a new Binary noise layer.
   *
   * @param value the value
   */
  public BinaryNoiseLayer(final double value) {
    super();
    setValue(value);
  }

  /**
   * Instantiates a new Binary noise layer.
   *
   * @param json the json
   */
  protected BinaryNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
    seed = json.get("seed").getAsLong();
    JsonElement enabled = json.get("enabled");
    //    this.enabled = enabled == null || enabled.getAsBoolean();
  }

  /**
   * Gets value.
   *
   * @return the value
   */
  public double getValue() {
    return value;
  }

  /**
   * Sets value.
   *
   * @param value the value
   */
  public void setValue(final double value) {
    this.value = value;
    shuffle(StochasticComponent.random.get().nextLong());
  }

  /**
   * From json binary noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the binary noise layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static BinaryNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BinaryNoiseLayer(json);
  }

  /**
   * Mask layer layer.
   *
   * @param density the density
   * @return the layer
   */
  @Nonnull
  public static Layer maskLayer(double density) {
    PipelineNetwork subnet = new PipelineNetwork(1);
    RefUtil.freeRef(subnet.add(new ProductInputsLayer(),
        subnet.add(new BinaryNoiseLayer(density), subnet.getInput(0)),
        subnet.getInput(0)));
    return subnet;
  }

  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    TensorArray data = fwd(dimensions, length);
    assert inputData.length() == data.length() : inputData.length() + " != " + data.length();
    inputData.freeRef();
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(input.getAccumulator());
    input.freeRef();
    return new Result(data, accumulator, alive);
  }

  /**
   * Clear.
   */
  public void clear() {
    final RefList<Tensor> maskList = this.maskList.addRef();
    synchronized (maskList) {
      maskList.clear();
    }
    maskList.freeRef();
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public void _free() {
    clear();
    maskList.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BinaryNoiseLayer addRef() {
    return (BinaryNoiseLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int[] dimensions, int length) {
    if (!maskList.isEmpty()) {
      Tensor temp_32_0004 = maskList.get(0);
      if (!RefArrays.equals(temp_32_0004.getDimensions(), dimensions)) {
        clear();
      }
      temp_32_0004.freeRef();
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
    tensorPrototype.freeRef();
    return new TensorArray(maskList.stream().limit(length).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private Result.Accumulator accumulator;

    /**
     * Instantiates a new Accumulator.
     *
     * @param accumulator the accumulator
     */
    public Accumulator(Result.Accumulator accumulator) {
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      this.accumulator.accept(buffer, new TensorArray(delta.stream().map(t -> {
        Tensor temp_32_0003 = t.map(x -> 0);
        t.freeRef();
        return temp_32_0003;
      }).toArray(Tensor[]::new)));
      delta.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
