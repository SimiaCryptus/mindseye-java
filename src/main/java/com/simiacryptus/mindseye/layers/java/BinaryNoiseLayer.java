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
import com.simiacryptus.lang.ref.*;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.*;

/**
 * The type Binary noise key.
 */
@SuppressWarnings("serial")
public class BinaryNoiseLayer extends LayerBase implements StochasticComponent {


  /**
   * The constant randomize.
   */
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
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
  List<Tensor> maskList = new ArrayList<>();
  private double value;
  private long seed = System.nanoTime();

  /**
   * Instantiates a new Binary noise key.
   */
  public BinaryNoiseLayer() {
    this(0.5);
  }

  /**
   * Instantiates a new Binary noise key.
   *
   * @param value the value
   */
  public BinaryNoiseLayer(final double value) {
    super();
    setValue(value);
  }

  /**
   * Instantiates a new Binary noise key.
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
   * From json binary noise key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the binary noise key
   */
  public static BinaryNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BinaryNoiseLayer(json);
  }

  @NotNull
  public static Layer maskLayer(double density) {
    PipelineNetwork subnet = new PipelineNetwork(1);
    subnet.wrap(new ProductInputsLayer(),
        subnet.wrap(new BinaryNoiseLayer(density), subnet.getInput(0)),
        subnet.getInput(0)
    ).freeRef();
    return subnet;
  }

  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
    TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (maskList.size() > 0 && !Arrays.equals(maskList.get(0).getDimensions(), dimensions)) {
      clear();
    }
    @Nonnull final Tensor tensorPrototype = new Tensor(dimensions);
    double amplitude = 1.0 / getValue();
    while (length > maskList.size()) {
      if(seed==0) {
        maskList.add(tensorPrototype.map(v -> amplitude, false));
      } else {
        Random random = new Random(seed * maskList.size());
        maskList.add(tensorPrototype.map(v -> random.nextDouble() < getValue() ? amplitude : 0, false));
      }
    }
    tensorPrototype.freeRef();
    TensorArray data = TensorArray.create(maskList.stream().limit(length).toArray(i -> new Tensor[i]));
    assert inputData.length() == data.length() : (inputData.length() + " != " + data.length());
    inputData.freeRef();
    return new Result(data, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      input.accumulate(buffer, TensorArray.wrap(delta.stream().map(t -> t.mapAndFree(x -> 0)).toArray(i -> new Tensor[i])));
      delta.freeRef();
    }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }


      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
    };
  }

  public void clear() {
    synchronized (maskList) {
      maskList.stream().forEach(ReferenceCounting::freeRef);
      maskList.clear();
    }
  }

  @Override
  protected void _free() {
    clear();
    super._free();
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
   * @return the value
   */
  @Nonnull
  public BinaryNoiseLayer setValue(final double value) {
    this.value = value;
    shuffle(StochasticComponent.random.get().nextLong());
    return this;
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
  public List<double[]> state() {
    return Arrays.asList();
  }

}
