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
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ImgBandScaleLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandScaleLayer.class);
  @Nullable
  private final double[] weights;

  protected ImgBandScaleLayer() {
    super();
    weights = null;
  }

  public ImgBandScaleLayer(@Nullable final double... bands) {
    super();
    weights = Arrays.copyOf(bands, bands.length);
  }

  protected ImgBandScaleLayer(@Nonnull final JsonObject json) {
    super(json);
    weights = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  @Nullable
  public double[] getWeights() {
    assert weights != null;
    if (!RefArrays.stream(weights).allMatch(Double::isFinite)) {
      throw new IllegalStateException(RefArrays.toString(weights));
    }
    return weights;
  }

  public void setWeights(@Nonnull IntToDoubleFunction f) {
    @Nullable final double[] bias = getWeights();
    assert bias != null;
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert RefArrays.stream(bias).allMatch(Double::isFinite);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandScaleLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandScaleLayer(json);
  }

  public void addWeights(@Nonnull DoubleSupplier f) {
    Util.add(f, getWeights());
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    Result result = eval(inObj[0].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    @Nullable final double[] weights = getWeights();
    final TensorList inData = input.getData();
    TensorArray data = fwd(weights, inData.addRef());
    boolean alive = input.isAlive();
    Accumulator accumulator = new Accumulator(inData, weights, getId(), isFrozen(), input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWeights()));
    return json;
  }

  public void set(@Nonnull double[] ds) {
    @Nullable final double[] bias = getWeights();
    for (int i = 0; i < ds.length; i++) {
      assert bias != null;
      bias[i] = ds[i];
    }
    assert bias != null;
    assert RefArrays.stream(bias).allMatch(Double::isFinite);
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(getWeights());
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandScaleLayer addRef() {
    return (ImgBandScaleLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(double[] weights, TensorList inData) {
    TensorArray tensorArray = new TensorArray(inData.stream().parallel().map(tensor -> {
      int[] dimensions = tensor.getDimensions();
      if (dimensions.length != 3) {
        tensor.freeRef();
        throw new IllegalArgumentException(
            RefArrays.toString(dimensions));
      }
      assert weights != null;
      if (dimensions[2] != weights.length) {
        tensor.freeRef();
        throw new IllegalArgumentException(RefString.format(
            "%s: %s does not have %s bands", getName(), RefArrays.toString(dimensions), weights.length));
      }
      return tensor.mapCoords(RefUtil.wrapInterface(c -> tensor.get(c) * weights[c.getCoords()[2]], tensor));
    }).toArray(Tensor[]::new));
    inData.freeRef();
    return tensorArray;
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList inData;
    private final double[] weights;
    private UUID id;
    private boolean frozen;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(TensorList inData, double[] weights, UUID id, boolean frozen, Result.Accumulator accumulator, boolean alive) {
      this.inData = inData;
      this.weights = weights;
      this.id = id;
      this.frozen = frozen;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (!frozen) {
        final Delta<UUID> deltaBuffer = buffer.get(id, weights);
        RefIntStream.range(0, delta.length()).forEach(RefUtil.wrapInterface(index -> {
          @Nonnull
          int[] dimensions = delta.getDimensions();
          int z = dimensions[2];
          int y = dimensions[1];
          int x = dimensions[0];
          final double[] array = RecycleBin.DOUBLES.obtain(z);
          Tensor deltaTensor = delta.get(index);
          @Nullable final double[] deltaArray = deltaTensor.getData();
          deltaTensor.freeRef();
          Tensor inputTensor = inData.get(index);
          @Nullable final double[] inputData = inputTensor.getData();
          inputTensor.freeRef();
          for (int i = 0; i < z; i++) {
            for (int j = 0; j < y * x; j++) {
              //array[i] += deltaArray[i + z * j];
              array[i] += deltaArray[i * x * y + j] * inputData[i * x * y + j];
            }
          }
          assert RefArrays.stream(array).allMatch(Double::isFinite);
          assert deltaBuffer != null;
          deltaBuffer.addInPlace(array);
          RecycleBin.DOUBLES.recycle(array, array.length);
        }, inData.addRef(), delta.addRef(), deltaBuffer));
      }
      if (alive) {
        try {
          this.accumulator.accept(buffer.addRef(), new TensorArray(delta.stream().map(t -> {
            return t.mapCoords(RefUtil.wrapInterface(c -> {
              assert weights != null;
              return t.get(c) * weights[c.getCoords()[2]];
            }, t));
          }).toArray(Tensor[]::new)));
        } finally {
          this.accumulator.freeRef();
        }
      }
      delta.freeRef();
      buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      inData.freeRef();
    }
  }
}
