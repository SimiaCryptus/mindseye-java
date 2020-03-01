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
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class ImgPixelGateLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelGateLayer.class);

  public ImgPixelGateLayer() {
    super();
  }

  protected ImgPixelGateLayer(@Nonnull final JsonObject json) {
    super(json);
  }

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

  @NotNull
  private TensorArray fwd(TensorList inputData, TensorList gateData, int[] inputDims) {
    return new TensorArray(RefIntStream.range(0, inputData.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
              Tensor inputTensor = inputData.get(i);
              Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
              Tensor outputTensor = new Tensor(inputDims[0], inputDims[1], inputDims[2]);
              outputTensor.setByCoord(RefUtil.wrapInterface((ToDoubleFunction<Coordinate>) c -> {
                    int[] coords = c.getCoords();
                    return inputTensor.get(coords[0], coords[1], coords[2])
                        * gateTensor.get(coords[0], coords[1], 0);
                  }, inputTensor, gateTensor));
              return outputTensor;
            }, inputData, gateData))
            .toArray(Tensor[]::new));
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

  private static class Accumulator extends Result.Accumulator {

    private final TensorList gateData;
    private final TensorList inputData;
    private final int[] inputDims;
    private Result.Accumulator inputAccumulator;
    private boolean inputAlive;
    private Result.Accumulator gateAccumulator;
    private boolean gateAlive;

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
