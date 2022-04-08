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

/**
 * This class is responsible for summing the pixels in an image.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class ImgPixelSumLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSumLayer.class);

  /**
   * Instantiates a new Img pixel sum layer.
   */
  public ImgPixelSumLayer() {
    super();
  }

  /**
   * Instantiates a new Img pixel sum layer.
   *
   * @param json the json
   */
  protected ImgPixelSumLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  /**
   * Returns a new {@link ImgPixelSumLayer} from a JSON object.
   *
   * @param json the JSON object to parse
   * @param rs   the resources to load
   * @return a new {@link ImgPixelSumLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPixelSumLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSumLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
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
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    TensorArray data = fwd(inputData, inputDims);
    boolean alive = input.isAlive();
    Accumulator accumulator = new Accumulator(inputDims, input.getAccumulator(), input.isAlive());
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
  ImgPixelSumLayer addRef() {
    return (ImgPixelSumLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList inputData, int[] inputDims) {
    TensorArray tensorArray = new TensorArray(inputData.stream().map(tensor -> {
      Tensor outputTensor = new Tensor(inputDims[0], inputDims[1], 1);
      outputTensor.setByCoord(RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputDims[2]).mapToDouble(RefUtil.wrapInterface(i -> {
          int[] coords = c.getCoords();
          return tensor.get(coords[0], coords[1], i);
        }, tensor == null ? null : tensor.addRef())).sum();
      }, tensor));
      return outputTensor;
    }).toArray(Tensor[]::new));
    inputData.freeRef();
    return tensorArray;
  }

  /**
   * The Accumulator class is used to track the results of an operation.
   *
   * @param inputDims   The dimensions of the input data.
   * @param accumulator The accumulator to use.
   * @param alive       A boolean value indicating whether the operation is still alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] inputDims;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inputDims   the input dims
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.inputDims = inputDims;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(delta.stream().map(deltaTensor -> {
          int[] deltaDims = deltaTensor.getDimensions();
          Tensor temp_47_0007 = new Tensor(deltaDims[0], deltaDims[1], inputDims[2]);
          temp_47_0007.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return deltaTensor.get(coords[0], coords[1], 0);
          }, deltaTensor.addRef()));
          Tensor temp_47_0003 = temp_47_0007.addRef();
          temp_47_0007.freeRef();
          deltaTensor.freeRef();
          return temp_47_0003;
        }).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
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
      accumulator.freeRef();
    }
  }
}
