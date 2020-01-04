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
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgBandScaleLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandScaleLayer.class);
  @Nullable
  private final double[] weights;

  protected ImgBandScaleLayer() {
    super();
    weights = null;
  }

  public ImgBandScaleLayer(@org.jetbrains.annotations.Nullable final double... bands) {
    super();
    weights = bands;
  }

  protected ImgBandScaleLayer(@Nonnull final JsonObject json) {
    super(json);
    weights = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  @Nullable
  public double[] getWeights() {
    if (!com.simiacryptus.ref.wrappers.RefArrays.stream(weights).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(com.simiacryptus.ref.wrappers.RefArrays.toString(weights));
    }
    return weights;
  }

  @Nonnull
  public ImgBandScaleLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    @Nullable
    final double[] bias = getWeights();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgBandScaleLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgBandScaleLayer(json);
  }

  @Nonnull
  public ImgBandScaleLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getWeights());
    return this;
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    return eval(inObj[0]);
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    @Nullable
    final double[] weights = getWeights();
    final TensorList inData = input.getData();
    @Nullable
    Function<Tensor, Tensor> tensorTensorFunction = tensor -> {
      if (tensor.getDimensions().length != 3) {
        throw new IllegalArgumentException(com.simiacryptus.ref.wrappers.RefArrays.toString(tensor.getDimensions()));
      }
      if (tensor.getDimensions()[2] != weights.length) {
        throw new IllegalArgumentException(String.format("%s: %s does not have %s bands", getName(),
            com.simiacryptus.ref.wrappers.RefArrays.toString(tensor.getDimensions()), weights.length));
      }
      return tensor.mapCoords(c -> tensor.get(c) * weights[c.getCoords()[2]]);
    };
    Tensor[] data = inData.stream().parallel().map(tensorTensorFunction).toArray(i -> new Tensor[i]);
    return new Result(new TensorArray(data),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (!isFrozen()) {
            final Delta<UUID> deltaBuffer = buffer.get(ImgBandScaleLayer.this.getId(), weights);
            com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length()).forEach(index -> {
              @Nonnull
              int[] dimensions = delta.getDimensions();
              int z = dimensions[2];
              int y = dimensions[1];
              int x = dimensions[0];
              final double[] array = RecycleBin.DOUBLES.obtain(z);
              Tensor deltaTensor = delta.get(index);
              @Nullable
              final double[] deltaArray = deltaTensor.getData();
              Tensor inputTensor = inData.get(index);
              @Nullable
              final double[] inputData = inputTensor.getData();
              for (int i = 0; i < z; i++) {
                for (int j = 0; j < y * x; j++) {
                  //array[i] += deltaArray[i + z * j];
                  array[i] += deltaArray[i * x * y + j] * inputData[i * x * y + j];
                }
              }
              assert com.simiacryptus.ref.wrappers.RefArrays.stream(array).allMatch(v -> Double.isFinite(v));
              deltaBuffer.addInPlace(array);
              RecycleBin.DOUBLES.recycle(array, array.length);
            });
          }
          if (input.isAlive()) {
            Tensor[] tensors = delta.stream().map(t -> {
              return t.mapCoords((c) -> t.get(c) * weights[c.getCoords()[2]]);
            }).toArray(i -> new Tensor[i]);
            @Nonnull
            TensorArray tensorArray = new TensorArray(tensors);
            input.accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWeights()));
    return json;
  }

  @Nonnull
  public Layer set(@Nonnull final double[] ds) {
    @Nullable
    final double[] bias = getWeights();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getWeights());
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgBandScaleLayer addRef() {
    return (ImgBandScaleLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgBandScaleLayer[] addRefs(ImgBandScaleLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandScaleLayer::addRef)
        .toArray((x) -> new ImgBandScaleLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgBandScaleLayer[][] addRefs(ImgBandScaleLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandScaleLayer::addRefs)
        .toArray((x) -> new ImgBandScaleLayer[x][]);
  }
}
