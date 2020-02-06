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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.function.IntFunction;

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

  public void setValue(double value) {
    this.value = value;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static DropoutNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new DropoutNoiseLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result inputResult = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputData = inputResult.getData();
    final int itemCnt = inputData.length();
    final Tensor[] mask = RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nonnull final Random random = new Random(seed);
          @Nullable final Tensor input = inputData.get(dataIndex);
          Tensor temp_36_0003 = input.map(x -> {
            if (seed == -1)
              return 1;
            return random.nextDouble() < getValue() ? 0 : 1.0 / getValue();
          });
          input.freeRef();
          return temp_36_0003;
        }, inputData.addRef())).toArray(Tensor[]::new);
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          RefUtil.addRefs(mask);
          inputResult.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (inputResult.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor deltaTensor = delta.get(dataIndex);
                  @Nullable final double[] deltaData = deltaTensor.getData();
                  @Nullable final double[] maskData = mask[dataIndex].getData();
                  @Nonnull final Tensor passback = new Tensor(deltaTensor.getDimensions());
                  deltaTensor.freeRef();
                  for (int i = 0; i < passback.length(); i++) {
                    passback.set(i, maskData[i] * deltaData[i]);
                  }
                  return passback;
                }, RefUtil.addRefs(mask), delta.addRef())).toArray(Tensor[]::new));
            inputResult.accumulate(buffer == null ? null : buffer.addRef(),
                tensorArray);
          }
          delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(mask);
          inputResult.freeRef();
        }
      };
      TensorArray data = new TensorArray(RefIntStream.range(0, itemCnt)
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            Tensor inputTensor = inputData.get(dataIndex);
            @Nullable final double[] input = inputTensor.getData();
            @Nullable final double[] maskT = mask[dataIndex].getData();
            @Nonnull final Tensor output = new Tensor(inputTensor.getDimensions());
            inputTensor.freeRef();
            @Nullable final double[] outputData = output.getData();
            for (int i = 0; i < outputData.length; i++) {
              outputData[i] = input[i] * maskT[i];
            }
            return output;
          }, RefUtil.addRefs(mask), inputData.addRef())).toArray(Tensor[]::new));
      return new Result(data, accumulator) {
        {
          inputResult.addRef();
        }

        @Override
        public boolean isAlive() {
          return inputResult.isAlive() || !isFrozen();
        }

        @Override
        public void _free() {
          inputResult.freeRef();
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(mask);
      inputData.freeRef();
      inputResult.freeRef();
    }
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
  DropoutNoiseLayer addRef() {
    return (DropoutNoiseLayer) super.addRef();
  }

}
