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
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * Class ImgBandBiasLayer
 *
 * @author
 * @version This class is used to ...
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  @Nullable
  private final double[] bias;

  /**
   * Instantiates a new Img band bias layer.
   */
  protected ImgBandBiasLayer() {
    super();
    bias = null;
  }

  /**
   * Instantiates a new Img band bias layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(final int bands) {
    super();
    bias = new double[bands];
  }

  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgBandBiasLayer(@Nonnull final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  /**
   * Returns the bias, or null if the bias is not set.
   *
   * @throws IllegalStateException if the bias is not set or if any of the bias values are not finite
   * @docgenVersion 9
   */
  @Nullable
  public double[] getBias() {
    assert bias != null;
    if (!RefArrays.stream(bias).allMatch(Double::isFinite)) {
      throw new IllegalStateException(RefArrays.toString(bias));
    }
    return bias;
  }

  /**
   * Sets the weights of the bias.
   *
   * @param f the function to apply to the bias.
   * @docgenVersion 9
   */
  public void setWeights(@Nonnull IntToDoubleFunction f) {
    @Nullable final double[] bias = getBias();
    assert bias != null;
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert RefArrays.stream(bias).allMatch(Double::isFinite);
  }

  /**
   * Sets the weights of the logarithm.
   *
   * @param value the value to set the weights to
   * @docgenVersion 9
   */
  public void setWeightsLog(double value) {
    assert bias != null;
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
  }

  /**
   * Creates a new {@link ImgBandBiasLayer} from the specified JSON object.
   *
   * @param json the JSON object to use
   * @param rs   the map of resources to use
   * @return the new {@link ImgBandBiasLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandBiasLayer(json);
  }

  /**
   * Adds the given input to the current instance.
   *
   * @param input the input to add
   * @return the resulting array
   * @throws NullPointerException if the input is null
   * @docgenVersion 9
   */
  @Nonnull
  public double[] add(@Nonnull final double[] input) {
    assert RefArrays.stream(input).allMatch(Double::isFinite);
    @Nullable final double[] bias = getBias();
    assert null != bias;
    if (input.length % bias.length != 0)
      throw new IllegalArgumentException();
    @Nonnull final double[] array = new double[input.length];
    final int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i / size];
    }
    assert RefArrays.stream(array).allMatch(Double::isFinite);
    return array;
  }

  /**
   * Adds the given weights to the bias.
   *
   * @param f the given weights
   * @docgenVersion 9
   */
  public void addWeights(@Nonnull DoubleSupplier f) {
    Util.add(f, getBias());
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    Result result = eval(inObj[0].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  /**
   * Evaluates the given input.
   *
   * @param input the input to evaluate
   * @return the result of the evaluation
   * @throws NullPointerException if the input is null
   * @docgenVersion 9
   */
  @Nonnull
  public Result eval(@Nonnull final Result input) {
    @Nullable final double[] bias = getBias();
    TensorArray data = fwd(bias, input.getData());
    boolean alive = input.isAlive();
    Accumulator accumulator = new Accumulator(bias, getId(), isFrozen(), input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }

  /**
   * Sets the bias of the neuron.
   *
   * @param ds the new bias
   * @throws NullPointerException     if ds is null
   * @throws IllegalArgumentException if ds is not finite
   * @docgenVersion 9
   */
  public void set(@Nonnull double[] ds) {
    @Nullable final double[] bias = getBias();
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
    return RefArrays.asList(getBias());
  }

  /**
   * Sets the given {@link Tensor} as this {@link Tensor}'s data.
   *
   * @param tensor The {@link Tensor} to set as this {@link Tensor}'s data
   * @docgenVersion 9
   */
  public void set(@Nonnull Tensor tensor) {
    set(tensor.getData());
    tensor.freeRef();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandBiasLayer addRef() {
    return (ImgBandBiasLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(double[] bias, TensorList inputData) {
    TensorArray tensorArray = new TensorArray(inputData.stream().parallel().map(r -> {
      int[] dimensions = r.getDimensions();
      if (dimensions.length != 3) {
        r.freeRef();
        throw new IllegalArgumentException(RefArrays.toString(dimensions));
      }
      assert bias != null;
      if (dimensions[2] != bias.length) {
        r.freeRef();
        throw new IllegalArgumentException(RefString.format(
            "%s: %s does not have %s bands", getName(), RefArrays.toString(dimensions), bias.length));
      }
      Tensor tensor = new Tensor(add(r.getData()), dimensions);
      r.freeRef();
      return tensor;
    }).toArray(Tensor[]::new));
    inputData.freeRef();
    return tensorArray;
  }

  /**
   * The Accumulator class represents an accumulator, which is used to accumulate bias values.
   *
   * @param bias        The bias values to accumulate.
   * @param id          The unique identifier for this accumulator.
   * @param frozen      Whether or not this accumulator is frozen.
   * @param accumulator The Result.Accumulator object for this accumulator.
   * @param alive       Whether or not this accumulator is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final double[] bias;
    private UUID id;
    private boolean frozen;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param bias        the bias
     * @param id          the id
     * @param frozen      the frozen
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(double[] bias, UUID id, boolean frozen, Result.Accumulator accumulator, boolean alive) {
      this.bias = bias;
      this.id = id;
      this.frozen = frozen;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (!frozen) {
        final Delta<UUID> deltaBuffer = buffer.get(id, bias);
        data.stream().parallel().forEach(RefUtil.wrapInterface((Consumer<? super Tensor>) d -> {
          assert bias != null;
          final double[] array = RecycleBin.DOUBLES.obtain(bias.length);
          @Nullable final double[] signal = d.getData();
          d.freeRef();
          final int size = signal.length / bias.length;
          for (int i = 0; i < signal.length; i++) {
            array[i / size] += signal[i];
            if (!Double.isFinite(array[i / size])) {
              array[i / size] = 0.0;
            }
          }
          assert RefArrays.stream(array).allMatch(Double::isFinite);
          assert deltaBuffer != null;
          deltaBuffer.addInPlace(array);
          RecycleBin.DOUBLES.recycle(array, array.length);
        }, deltaBuffer == null ? null : deltaBuffer.addRef()));
        if (null != deltaBuffer)
          deltaBuffer.freeRef();
      }
      if (alive) {
        this.accumulator.accept(buffer, data);
      } else {
        data.freeRef();
        buffer.freeRef();
      }
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
