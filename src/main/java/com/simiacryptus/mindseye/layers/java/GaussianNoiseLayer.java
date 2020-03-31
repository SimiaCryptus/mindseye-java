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
import com.simiacryptus.mindseye.layers.StochasticComponent;
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
import java.util.Random;
import java.util.UUID;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class GaussianNoiseLayer extends LayerBase implements StochasticComponent {

  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Nonnull
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
    seed = json.get("seed").getAsLong();
  }

  public double getValue() {
    return value;
  }

  public void setValue(double value) {
    this.value = value;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static GaussianNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GaussianNoiseLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = in0.getData();
    int[] dimensions = inputData.getDimensions();
    TensorArray data = fwd(inputData);
    boolean alive = in0.isAlive();
    Result.Accumulator accumulator = new Accumulator(dimensions, in0.getAccumulator(), in0.isAlive());
    in0.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    json.addProperty("seed", seed);
    return json;
  }

  public void shuffle() {
    shuffle(GaussianNoiseLayer.random.get().nextLong());
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  @Override
  public void shuffle(long seed) {
    this.seed = seed;
  }

  @Override
  public void clearNoise() {
    this.seed = 0;
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  GaussianNoiseLayer addRef() {
    return (GaussianNoiseLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inputData) {
    final int itemCnt = inputData.length();
    return new TensorArray(RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nonnull final Random random1 = new Random(seed);
          Tensor tensor = inputData.get(dataIndex);
          @Nullable final Tensor input = tensor.copy();
          tensor.freeRef();
          for (int i = 0; i < input.length(); i++) {
            input.set(i, input.get(i) + random1.nextGaussian() * getValue());
          }
          return input;
        }, inputData)).toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final int[] dimensions;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(int[] dimensions, Result.Accumulator accumulator, boolean alive) {
      this.dimensions = dimensions;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              Tensor tensor = delta.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(dimensions);
              passback.set((IntToDoubleFunction) tensor::get);
              tensor.freeRef();
              return passback;
            }, delta)).toArray(Tensor[]::new));
        this.accumulator.accept(buffer, tensorArray);
      } else {
        delta.freeRef();
        buffer.freeRef();
      }
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
