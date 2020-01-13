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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
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
import java.util.function.Consumer;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase {

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
    if (!RefArrays.stream(bias).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(RefArrays.toString(bias));
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
    assert RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this.addRef();
  }

  @Nonnull
  public ImgBandBiasLayer setWeightsLog(final double value) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json);
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[] addRefs(ImgBandBiasLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRef)
        .toArray((x) -> new ImgBandBiasLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgBandBiasLayer[][] addRefs(ImgBandBiasLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgBandBiasLayer::addRefs)
        .toArray((x) -> new ImgBandBiasLayer[x][]);
  }

  @Nonnull
  public double[] add(@Nonnull final double[] input) {
    assert RefArrays.stream(input).allMatch(v -> Double.isFinite(v));
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
    assert RefArrays.stream(array).allMatch(v -> Double.isFinite(v));
    return array;
  }

  @Nonnull
  public ImgBandBiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
    return this.addRef();
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    Result temp_24_0005 = eval(inObj[0].addRef());
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    return temp_24_0005;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    @Nullable
    final double[] bias = getBias();
    final ImgBandBiasLayer imgBandBiasLayer = ImgBandBiasLayer.this.addRef();
    try {
      try {
        TensorList temp_24_0009 = input.getData();
        Result temp_24_0008 = new Result(new TensorArray(temp_24_0009.stream().parallel().map(r -> {
          if (r.getDimensions().length != 3) {
            IllegalArgumentException temp_24_0003 = new IllegalArgumentException(RefArrays.toString(r.getDimensions()));
            if (null != r)
              r.freeRef();
            throw temp_24_0003;
          }
          if (r.getDimensions()[2] != bias.length) {
            IllegalArgumentException temp_24_0004 = new IllegalArgumentException(RefString.format(
                "%s: %s does not have %s bands", getName(), RefArrays.toString(r.getDimensions()), bias.length));
            if (null != r)
              r.freeRef();
            throw temp_24_0004;
          }
          Tensor temp_24_0002 = new Tensor(add(r.getData()), r.getDimensions());
          if (null != r)
            r.freeRef();
          return temp_24_0002;
        }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
          {
            input.addRef();
            imgBandBiasLayer.addRef();
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            if (!ImgBandBiasLayer.this.isFrozen()) {
              final Delta<UUID> deltaBuffer = buffer.get(imgBandBiasLayer.getId(), bias);
              data.stream().parallel().forEach(RefUtil.wrapInterface((Consumer<? super Tensor>) d -> {
                final double[] array = RecycleBin.DOUBLES.obtain(bias.length);
                @Nullable
                final double[] signal = d.getData();
                if (null != d)
                  d.freeRef();
                final int size = signal.length / bias.length;
                for (int i = 0; i < signal.length; i++) {
                  array[i / size] += signal[i];
                  if (!Double.isFinite(array[i / size])) {
                    array[i / size] = 0.0;
                  }
                }
                assert RefArrays.stream(array).allMatch(v -> Double.isFinite(v));
                deltaBuffer.addInPlace(array);
                RecycleBin.DOUBLES.recycle(array, array.length);
              }, deltaBuffer == null ? null : deltaBuffer.addRef()));
              if (null != deltaBuffer)
                deltaBuffer.freeRef();
            }
            if (input.isAlive()) {
              input.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
            }
            if (null != data)
              data.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused") void _free() {
            input.freeRef();
            imgBandBiasLayer.freeRef();
          }
        }) {

          {
            input.addRef();
          }

          @Override
          public boolean isAlive() {
            return input.isAlive() || !isFrozen();
          }

          public void _free() {
            input.freeRef();
          }
        };
        if (null != temp_24_0009)
          temp_24_0009.freeRef();
        return temp_24_0008;
      } finally {
        input.freeRef();
      }
    } finally {
      if (null != imgBandBiasLayer)
        imgBandBiasLayer.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
    assert RefArrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this.addRef();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(getBias());
  }

  public ImgBandBiasLayer set(final Tensor tensor) {
    ImgBandBiasLayer temp_24_0007 = (ImgBandBiasLayer) set(tensor.getData());
    if (null != tensor)
      tensor.freeRef();
    return temp_24_0007;
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgBandBiasLayer addRef() {
    return (ImgBandBiasLayer) super.addRef();
  }
}
