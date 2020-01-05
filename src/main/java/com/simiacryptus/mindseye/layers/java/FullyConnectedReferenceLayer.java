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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public @RefAware
class FullyConnectedReferenceLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedReferenceLayer.class);
  @Nullable
  public final int[] inputDims;
  @Nullable
  public final int[] outputDims;
  @Nullable
  public final Tensor weights;

  protected FullyConnectedReferenceLayer() {
    super();
    outputDims = null;
    weights = null;
    inputDims = null;
  }

  public FullyConnectedReferenceLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int inputs = Tensor.length(inputDims);
    final int outputs = Tensor.length(outputDims);
    weights = new Tensor(inputs, outputs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outputs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  protected FullyConnectedReferenceLayer(@Nonnull final JsonObject json,
                                         Map<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  @Nullable
  public Tensor getWeights() {
    return weights;
  }

  @Nonnull
  public FullyConnectedReferenceLayer setByCoord(@Nonnull final ToDoubleFunction<Coordinate> f) {
    weights.coordStream(true).forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  @Nonnull
  public FullyConnectedReferenceLayer setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream(true).forEach(in -> {
      new Tensor(outputDims).coordStream(true).forEach(out -> {
        weights.set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
      });
    });
    return this;
  }

  @Nonnull
  public FullyConnectedReferenceLayer setWeightsLog(final double value) {
    weights.coordStream(false).forEach(c -> {
      weights.set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }

  @SuppressWarnings("unused")
  public static FullyConnectedReferenceLayer fromJson(@Nonnull final JsonObject json,
                                                      Map<CharSequence, byte[]> rs) {
    return new FullyConnectedReferenceLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  FullyConnectedReferenceLayer[] addRefs(
      FullyConnectedReferenceLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayer::addRef)
        .toArray((x) -> new FullyConnectedReferenceLayer[x]);
  }

  public static @SuppressWarnings("unused")
  FullyConnectedReferenceLayer[][] addRefs(
      FullyConnectedReferenceLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayer::addRefs)
        .toArray((x) -> new FullyConnectedReferenceLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result inputResult = inObj[0];
    final TensorList indata = inputResult.getData();
    @Nonnull
    int[] inputDimensions = indata.getDimensions();
    assert Tensor.length(inputDimensions) == Tensor.length(this.inputDims) : RefArrays
        .toString(inputDimensions) + " == " + RefArrays.toString(this.inputDims);
    return new Result(
        new TensorArray(RefIntStream.range(0, indata.length()).mapToObj(index -> {
          @Nullable final Tensor input = indata.get(index);
          @Nullable final Tensor output = new Tensor(outputDims);
          weights.coordStream(false).forEach(c -> {
            int[] coords = c.getCoords();
            double prev = output.get(coords[1]);
            double w = weights.get(c);
            double i = input.get(coords[0]);
            double value = prev + w * i;
            output.set(coords[1], value);
          });
          return output;
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        Tensor[] array = RefIntStream.range(0, indata.length()).mapToObj(i -> {
          @Nullable final Tensor inputTensor = indata.get(i);
          @Nullable final Tensor deltaTensor = delta.get(i);
          @Nonnull
          Tensor weights = new Tensor(FullyConnectedReferenceLayer.this.weights.getDimensions());
          weights.coordStream(false).forEach(c -> {
            int[] coords = c.getCoords();
            weights.set(c, inputTensor.get(coords[0]) * deltaTensor.get(coords[1]));
          });
          return weights;
        }).toArray(i -> new Tensor[i]);
        Tensor tensor = RefArrays.stream(array).reduce((a, b) -> {
          return a.addAndFree(b);
        }).get();
        buffer.get(this.getId(), weights.getData()).addInPlace(tensor.getData());
      }
      if (inputResult.isAlive()) {
        @Nonnull final TensorList tensorList = new TensorArray(
            RefIntStream.range(0, indata.length()).mapToObj(i -> {
              @Nullable final Tensor inputTensor = new Tensor(inputDims);
              @Nullable final Tensor deltaTensor = delta.get(i);
              weights.coordStream(false).forEach(c -> {
                int[] coords = c.getCoords();
                inputTensor.set(coords[0],
                    inputTensor.get(coords[0]) + weights.get(c) * deltaTensor.get(coords[1]));
              });
              return inputTensor;
            }).toArray(i -> new Tensor[i]));
        inputResult.accumulate(buffer, tensorList);
      }
    }) {

      @Override
      public boolean isAlive() {
        return inputResult.isAlive() || !isFrozen();
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  public void set(@Nonnull final DoubleSupplier f) {
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(@Nonnull final IntToDoubleFunction f) {
    weights.set(f);
    return this;
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(final double[] data) {
    weights.set(data);
    return this;
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(@Nonnull final Tensor data) {
    weights.set(data);
    return this;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(getWeights().getData());
  }

  public void _free() {
    super._free();
  }

  public @Override
  @SuppressWarnings("unused")
  FullyConnectedReferenceLayer addRef() {
    return (FullyConnectedReferenceLayer) super.addRef();
  }

}
