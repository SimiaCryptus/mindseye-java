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
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * ImgPixelGateLayer class
 *
 * @author Author Name
 * @version 1.0
 * @docgenVersion 9
 * @since 1.0
 */
@SuppressWarnings("serial")
public class ImgPixelGateLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelGateLayer.class);

  /**
   * Instantiates a new Img pixel gate layer.
   */
  public ImgPixelGateLayer() {
    super();
  }

  /**
   * Instantiates a new Img pixel gate layer.
   *
   * @param json the json
   */
  protected ImgPixelGateLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  /**
   * @param json the JSON object to use for creating the {@link ImgPixelGateLayer}
   * @param rs   the map of resources to use
   * @return a new {@link ImgPixelGateLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPixelGateLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelGateLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 2 == inObj.length;
    Result result = eval(inObj[0].addRef(), inObj[1].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  /**
   * @param input the input to the evaluation
   * @param gate  the gate to use in the evaluation
   * @return the result of the evaluation
   * @throws NullPointerException if either input or gate is null
   * @docgenVersion 9
   */
  @Nonnull
  public Result eval(@Nonnull final Result input, @Nonnull final Result gate) {
    final TensorList inputData = input.getData();
    final TensorList gateData = gate.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    TensorArray data = fwd(inputData.addRef(), gateData.addRef(), inputDims);
    boolean alive = input.isAlive();
    Accumulator accumulator = new Accumulator(gateData, inputData, inputDims, input.getAccumulator(), input.isAlive(), gate.getAccumulator(), gate.isAlive());
    gate.freeRef();
    input.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
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
  ImgPixelGateLayer addRef() {
    return (ImgPixelGateLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inputData, TensorList gateData, int[] inputDims) {
    return new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
          Tensor inputTensor = inputData.get(i);
          Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
          Tensor outputTensor = new Tensor(inputDims[0], inputDims[1], inputDims[2]);
          outputTensor.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return inputTensor.get(coords[0], coords[1], coords[2])
                * gateTensor.get(coords[0], coords[1], 0);
          }, inputTensor, gateTensor));
          return outputTensor;
        }, inputData, gateData))
        .toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to accumulate gate and input data.
   *
   * @author John Doe
   * @version 1.0
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final TensorList gateData;
    private final TensorList inputData;
    private final int[] inputDims;
    private Result.Accumulator inputAccumulator;
    private boolean inputAlive;
    private Result.Accumulator gateAccumulator;
    private boolean gateAlive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param gateData         the gate data
     * @param inputData        the input data
     * @param inputDims        the input dims
     * @param inputAccumulator the input accumulator
     * @param inputAlive       the input alive
     * @param gateAccumulator  the gate accumulator
     * @param gateAlive        the gate alive
     */
    public Accumulator(TensorList gateData, TensorList inputData, int[] inputDims, Result.Accumulator inputAccumulator, boolean inputAlive, Result.Accumulator gateAccumulator, boolean gateAlive) {
      this.gateData = gateData;
      this.inputData = inputData;
      this.inputDims = inputDims;
      this.inputAccumulator = inputAccumulator;
      this.inputAlive = inputAlive;
      this.gateAccumulator = gateAccumulator;
      this.gateAlive = gateAlive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (inputAlive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                  Tensor deltaTensor = delta.get(i);
                  Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
                  Tensor feedbackTensor = new Tensor(inputData.getDimensions());
                  feedbackTensor.setByCoord(RefUtil.wrapInterface(c -> {
                        int[] coords = c.getCoords();
                        return deltaTensor.get(coords[0], coords[1], coords[2])
                            * gateTensor.get(coords[0], coords[1], 0);
                      }, gateTensor.addRef(),
                      deltaTensor.addRef()));
                  gateTensor.freeRef();
                  deltaTensor.freeRef();
                  return feedbackTensor;
                }, delta.addRef(), inputData.addRef(),
                gateData.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        inputAccumulator.accept(buffer1, tensorArray);
      }
      if (gateAlive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                  Tensor deltaTensor = delta.get(i);
                  Tensor inputTensor = inputData.get(i);
                  Tensor feedbackTensor = new Tensor(gateData.getDimensions());
                  feedbackTensor.setByCoord(RefUtil.wrapInterface(
                      c -> RefIntStream.range(0, inputDims[2]).mapToDouble(RefUtil.wrapInterface(b -> {
                            int[] coords = c.getCoords();
                            return deltaTensor.get(coords[0], coords[1], b)
                                * inputTensor.get(coords[0], coords[1], b);
                          }, inputTensor.addRef(),
                          deltaTensor.addRef())).sum(),
                      inputTensor.addRef(),
                      deltaTensor.addRef()));
                  inputTensor.freeRef();
                  deltaTensor.freeRef();
                  return feedbackTensor;
                }, gateData.addRef(), delta.addRef(),
                inputData.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        gateAccumulator.accept(buffer1, tensorArray);
      }
      delta.freeRef();
      if (null != buffer)
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
      gateAccumulator.freeRef();
      inputAccumulator.freeRef();
      gateData.freeRef();
      inputData.freeRef();
    }
  }
}
