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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class DropoutNoiseLayer extends LayerBase implements StochasticComponent {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DropoutNoiseLayer.class);
  long seed = StochasticComponent.random.get().nextLong();
  private double value;

  public DropoutNoiseLayer() {
    this(0.5);
  }

  public DropoutNoiseLayer(final double value) {
    super();
    setValue(value);
  }

  protected DropoutNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
  }

  public double getValue() {
    return value;
  }

  @Nonnull
  public DropoutNoiseLayer setValue(final double value) {
    this.value = value;
    return this;
  }

  @SuppressWarnings("unused")
  public static DropoutNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new DropoutNoiseLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    final int itemCnt = inputData.length();
    final Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nonnull final Random random = new Random(seed);
      @Nullable final Tensor input = inputData.get(dataIndex);
      return input.map(x -> {
        if (seed == -1)
          return 1;
        return random.nextDouble() < getValue() ? 0 : (1.0 / getValue());
      });
    }).toArray(i -> new Tensor[i]);
    return new Result(new TensorArray(IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      Tensor inputTensor = inputData.get(dataIndex);
      @Nullable final double[] input = inputTensor.getData();
      @Nullable final double[] maskT = mask[dataIndex].getData();
      @Nonnull final Tensor output = new Tensor(inputTensor.getDimensions());
      @Nullable final double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (inputResult.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nullable final double[] maskData = mask[dataIndex].getData();
          @Nonnull final Tensor passback = new Tensor(deltaTensor.getDimensions());
          for (int i = 0; i < passback.length(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inputResult.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return inputResult.isAlive() || !isFrozen();
      }

      @Override
      protected void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }

  @Override
  public void shuffle(final long seed) {
    //log.info(String.format("Set %s to random seed %s", getName(), seed));
    this.seed = StochasticComponent.random.get().nextLong();
  }

  @Override
  public void clearNoise() {
    //log.info(String.format("Set %s to random null seed", getName()));
    seed = -1;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
