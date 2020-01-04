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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class ImgPixelSumLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSumLayer.class);

  public ImgPixelSumLayer() {
    super();
  }

  protected ImgPixelSumLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @SuppressWarnings("unused")
  public static ImgPixelSumLayer fromJson(@Nonnull final JsonObject json,
                                          com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ImgPixelSumLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgPixelSumLayer[] addRefs(ImgPixelSumLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSumLayer::addRef)
        .toArray((x) -> new ImgPixelSumLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgPixelSumLayer[][] addRefs(ImgPixelSumLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSumLayer::addRefs)
        .toArray((x) -> new ImgPixelSumLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 1 == inObj.length;
    return eval(inObj[0]);
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    return new Result(new TensorArray(inputData.stream().map(tensor -> {
      return new Tensor(inputDims[0], inputDims[1], 1).setByCoord(c -> {
        return com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDims[2]).mapToDouble(i -> {
          int[] coords = c.getCoords();
          return tensor.get(coords[0], coords[1], i);
        }).sum();
      });
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(delta.stream().map(deltaTensor -> {
          int[] deltaDims = deltaTensor.getDimensions();
          return new Tensor(deltaDims[0], deltaDims[1], inputDims[2]).setByCoord(c -> {
            int[] coords = c.getCoords();
            return deltaTensor.get(coords[0], coords[1], 0);
          });
        }).toArray(i -> new Tensor[i]));
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
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgPixelSumLayer addRef() {
    return (ImgPixelSumLayer) super.addRef();
  }
}
