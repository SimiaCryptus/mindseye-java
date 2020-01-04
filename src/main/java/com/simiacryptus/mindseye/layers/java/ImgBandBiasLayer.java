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
import com.simiacryptus.util.FastRandom;
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
import java.util.function.IntToDoubleFunction;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class ImgBandBiasLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  @Nullable
  private final double[] bias;

  protected ImgBandBiasLayer() {
    super();
    bias = null;
  }

  public ImgBandBiasLayer(final int bands) {
    super();
    bias = new double[bands];
  }

  protected ImgBandBiasLayer(@Nonnull final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  @Nullable
  public double[] getBias() {
    if (!com.simiacryptus.ref.wrappers.RefArrays.stream(bias).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(com.simiacryptus.ref.wrappers.RefArrays.toString(bias));
    }
    return bias;
  }

  @Nonnull
  public ImgBandBiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    @Nullable
    final double[] bias = getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }

  @Nonnull
  public ImgBandBiasLayer setWeightsLog(final double value) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
    return this;
  }

  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json);
  }

  @Nonnull
  public double[] add(@Nonnull final double[] input) {
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(input).allMatch(v -> Double.isFinite(v));
    assert null != input;
    @Nullable
    final double[] bias = getBias();
    assert null != bias;
    if (input.length % bias.length != 0)
      throw new IllegalArgumentException();
    @Nonnull
    final double[] array = new double[input.length];
    final int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i / size];
    }
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(array).allMatch(v -> Double.isFinite(v));
    return array;
  }

  @Nonnull
  public ImgBandBiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
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
    final double[] bias = getBias();
    return new Result(new TensorArray(input.getData().stream().parallel().map(r -> {
      if (r.getDimensions().length != 3) {
        throw new IllegalArgumentException(com.simiacryptus.ref.wrappers.RefArrays.toString(r.getDimensions()));
      }
      if (r.getDimensions()[2] != bias.length) {
        throw new IllegalArgumentException(String.format("%s: %s does not have %s bands", getName(),
            com.simiacryptus.ref.wrappers.RefArrays.toString(r.getDimensions()), bias.length));
      }
      return new Tensor(add(r.getData()), r.getDimensions());
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (!isFrozen()) {
        final Delta<UUID> deltaBuffer = buffer.get(ImgBandBiasLayer.this.getId(), bias);
        data.stream().parallel().forEach(d -> {
          final double[] array = RecycleBin.DOUBLES.obtain(bias.length);
          @Nullable
          final double[] signal = d.getData();
          final int size = signal.length / bias.length;
          for (int i = 0; i < signal.length; i++) {
            array[i / size] += signal[i];
            if (!Double.isFinite(array[i / size])) {
              array[i / size] = 0.0;
            }
          }
          assert com.simiacryptus.ref.wrappers.RefArrays.stream(array).allMatch(v -> Double.isFinite(v));
          deltaBuffer.addInPlace(array);
          RecycleBin.DOUBLES.recycle(array, array.length);
        });
      }
      if (input.isAlive()) {
        input.accumulate(buffer, data);
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
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }

  @Nonnull
  public Layer set(@Nonnull final double[] ds) {
    @Nullable
    final double[] bias = getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList(getBias());
  }

  public ImgBandBiasLayer set(final Tensor tensor) {
    return (ImgBandBiasLayer) set(tensor.getData());
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgBandBiasLayer addRef() {
    return (ImgBandBiasLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[] addRefs(ImgBandBiasLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRef)
        .toArray((x) -> new ImgBandBiasLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[][] addRefs(ImgBandBiasLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRefs)
        .toArray((x) -> new ImgBandBiasLayer[x][]);
  }
}
