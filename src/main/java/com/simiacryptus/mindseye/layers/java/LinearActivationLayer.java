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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * This class represents a linear activation layer.
 *
 * @author John Doe
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class LinearActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  @Nullable
  private final Tensor weights;

  /**
   * Instantiates a new Linear activation layer.
   */
  public LinearActivationLayer() {
    this(1.);
  }

  /**
   * Instantiates a new Linear activation layer.
   *
   * @param scale the scale
   */
  public LinearActivationLayer(double scale) {
    this(scale, 0.);
  }

  /**
   * Instantiates a new Linear activation layer.
   *
   * @param scale the scale
   * @param bias  the bias
   */
  public LinearActivationLayer(double scale, double bias) {
    super();
    weights = new Tensor(2);
    weights.set(0, scale);
    weights.set(1, bias);
  }

  /**
   * Instantiates a new Linear activation layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected LinearActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }

  /**
   * Returns the bias of the neuron.
   *
   * @return the bias of the neuron
   * @docgenVersion 9
   */
  public double getBias() {
    assert weights != null;
    return weights.get(1);
  }

  /**
   * Sets the bias for the perceptron.
   *
   * @param bias the new bias
   * @throws IllegalArgumentException if the bias is not finite
   * @docgenVersion 9
   */
  public void setBias(double bias) {
    if (!Double.isFinite(bias))
      throw new IllegalArgumentException();
    assert weights != null;
    weights.set(1, bias);
  }

  @Nullable
  @Override
  public String getName() {
    String eqStr = isFrozen() ? "== " : "=> ";
    assert weights != null;
    if (weights.get(0) == 1.0) {
      return eqStr + RefString.format("x + %.1e", weights.get(1)) + (isFrozen() ? "" : "!");
    } else if (weights.get(1) == 0.0) {
      return eqStr + RefString.format("%.1e x", weights.get(0)) + (isFrozen() ? "" : "!");
    } else {
      return eqStr + RefString.format("%.1e x + %.1e", weights.get(0), weights.get(1));
    }
  }

  /**
   * Returns the scale.
   *
   * @return the scale
   * @docgenVersion 9
   */
  public double getScale() {
    assert weights != null;
    return weights.get(0);
  }

  /**
   * Sets the scale.
   *
   * @param scale the scale
   * @throws IllegalArgumentException if the scale is not finite
   * @docgenVersion 9
   */
  public void setScale(double scale) {
    if (!Double.isFinite(scale))
      throw new IllegalArgumentException();
    assert weights != null;
    weights.set(0, scale);
  }

  /**
   * Creates a new {@link LinearActivationLayer} from a JSON object.
   *
   * @param json the JSON object to use
   * @param rs   the map of character sequences to byte arrays
   * @return the new {@link LinearActivationLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static LinearActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LinearActivationLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    boolean alive = in0.isAlive();
    TensorArray data = fwd(in0.getData());
    Result.Accumulator accumulator = new Accumulator(in0.getData(), weights.addRef(), getId(), isFrozen(), in0.getAccumulator(), alive);
    in0.freeRef();
    return new Result(data, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    assert weights != null;
    return RefArrays.asList(weights.getData());
  }

  /**
   * Frees resources used by this object.
   *
   * @docgenVersion 9
   */
  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LinearActivationLayer addRef() {
    return (LinearActivationLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inData) {
    final int itemCnt = inData.length();
    assert weights != null;
    final double scale = weights.get(0);
    final double bias = weights.get(1);
    return new TensorArray(RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor inputTensor = inData.get(dataIndex);
          Tensor outputTensor = inputTensor.map(v -> {
            final double r = scale * v + bias;
            return Double.isFinite(r) ? r : 0;
          });
          inputTensor.freeRef();
          return outputTensor;
        }, inData)).toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class represents an accumulator, which is used to accumulate the results of a computation.
   *
   * @param inData      The input data to be accumulated.
   * @param weights     The weights to be used in the accumulation.
   * @param id          The unique identifier of the accumulator.
   * @param frozen      Whether or not the accumulator is frozen.
   * @param accumulator The Result.Accumulator object used to accumulate the results.
   * @param alive       Whether or not the accumulator is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList inData;
    private Tensor weights;
    private UUID id;
    private boolean frozen;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inData      the in data
     * @param weights     the weights
     * @param id          the id
     * @param frozen      the frozen
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(TensorList inData, Tensor weights, UUID id, boolean frozen, Result.Accumulator accumulator, boolean alive) {
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
        RefIntStream.range(0, delta.length()).forEach(RefUtil.wrapInterface(dataIndex -> {
          @Nullable
          Tensor deltaT = delta.get(dataIndex);
          @Nullable
          Tensor inputT = inData.get(dataIndex);
          @Nullable final double[] deltaData = deltaT.getData();
          @Nullable final double[] inputData = inputT.getData();
          @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            weightDelta.add(0, deltaData[i] * inputData[inputData.length == 1 ? 0 : i]);
            weightDelta.add(1, deltaData[i]);
          }
          deltaT.freeRef();
          inputT.freeRef();
          Delta<UUID> temp_04_0006 = buffer.get(id, weights.getData());
          assert temp_04_0006 != null;
          temp_04_0006.addInPlace(weightDelta.getData());
          temp_04_0006.freeRef();
          weightDelta.freeRef();
        }, buffer.addRef(), inData.addRef(), weights.addRef(), delta.addRef()));
      }
      if (alive) {
        @Nonnull final TensorList tensorList = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nullable
              Tensor tensor = delta.get(dataIndex);
              @Nullable final double[] deltaData = tensor.getData();
              tensor.freeRef();
              @Nonnull final Tensor passback = new Tensor(inData.getDimensions());
              for (int i = 0; i < passback.length(); i++) {
                passback.set(i, deltaData[i] * weights.get(0));
              }
              return passback;
            }, inData.addRef(), delta.addRef()))
            .toArray(Tensor[]::new));
        this.accumulator.accept(buffer.addRef(), tensorList);
      }
      delta.freeRef();
      buffer.freeRef();
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      inData.freeRef();
      weights.freeRef();
      accumulator.freeRef();
    }
  }
}
