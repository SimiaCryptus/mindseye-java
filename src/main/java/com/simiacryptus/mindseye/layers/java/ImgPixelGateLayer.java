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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class ImgPixelGateLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelGateLayer.class);

  public ImgPixelGateLayer() {
    super();
  }

  protected ImgPixelGateLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @SuppressWarnings("unused")
  public static ImgPixelGateLayer fromJson(@Nonnull final JsonObject json,
                                           Map<CharSequence, byte[]> rs) {
    return new ImgPixelGateLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgPixelGateLayer[] addRefs(ImgPixelGateLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRef)
        .toArray((x) -> new ImgPixelGateLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgPixelGateLayer[][] addRefs(ImgPixelGateLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRefs)
        .toArray((x) -> new ImgPixelGateLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 2 == inObj.length;
    return eval(inObj[0], inObj[1]);
  }

  @Nonnull
  public Result eval(@Nonnull final Result input, @Nonnull final Result gate) {
    final TensorList inputData = input.getData();
    final TensorList gateData = gate.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    return new Result(
        new TensorArray(RefIntStream.range(0, inputData.length()).mapToObj(i -> {
          Tensor inputTensor = inputData.get(i);
          Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
          return new Tensor(inputDims[0], inputDims[1], inputDims[2]).setByCoord(c -> {
            int[] coords = c.getCoords();
            return inputTensor.get(coords[0], coords[1], coords[2]) * gateTensor.get(coords[0], coords[1], 0);
          });
        }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            RefIntStream.range(0, delta.length()).mapToObj(i -> {
              Tensor deltaTensor = delta.get(i);
              Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
              return new Tensor(input.getData().getDimensions()).setByCoord(c -> {
                int[] coords = c.getCoords();
                return deltaTensor.get(coords[0], coords[1], coords[2]) * gateTensor.get(coords[0], coords[1], 0);
              });
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
      if (gate.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            RefIntStream.range(0, delta.length()).mapToObj(i -> {
              Tensor deltaTensor = delta.get(i);
              Tensor inputTensor = inputData.get(i);
              return new Tensor(gateData.getDimensions()).setByCoord(
                  c -> RefIntStream.range(0, inputDims[2]).mapToDouble(b -> {
                    int[] coords = c.getCoords();
                    return deltaTensor.get(coords[0], coords[1], b) * inputTensor.get(coords[0], coords[1], b);
                  }).sum());
            }).toArray(i -> new Tensor[i]));
        gate.accumulate(buffer, tensorArray);
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
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgPixelGateLayer addRef() {
    return (ImgPixelGateLayer) super.addRef();
  }
}
