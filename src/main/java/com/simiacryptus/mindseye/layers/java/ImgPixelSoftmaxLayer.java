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
class ImgPixelSoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSoftmaxLayer.class);

  public ImgPixelSoftmaxLayer() {
    super();
  }

  protected ImgPixelSoftmaxLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @SuppressWarnings("unused")
  public static ImgPixelSoftmaxLayer fromJson(@Nonnull final JsonObject json,
                                              Map<CharSequence, byte[]> rs) {
    return new ImgPixelSoftmaxLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgPixelSoftmaxLayer[] addRefs(ImgPixelSoftmaxLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSoftmaxLayer::addRef)
        .toArray((x) -> new ImgPixelSoftmaxLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgPixelSoftmaxLayer[][] addRefs(ImgPixelSoftmaxLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSoftmaxLayer::addRefs)
        .toArray((x) -> new ImgPixelSoftmaxLayer[x][]);
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
    final int inputBands = inputDims[2];
    final int width = inputDims[0];
    final int height = inputDims[1];
    TensorArray maxima = new TensorArray(inputData.stream().map(inputTensor -> {
      return new Tensor(width, height, 1).setByCoord(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(band -> {
          int[] coords = c.getCoords();
          return inputTensor.get(coords[0], coords[1], band);
        }).max().getAsDouble();
      });
    }).toArray(i -> new Tensor[i]));
    TensorArray exps = new TensorArray(
        RefIntStream.range(0, inputData.length()).mapToObj(index -> {
          final Tensor inputTensor = inputData.get(index);
          Tensor maxTensor = maxima.get(index);
          return new Tensor(inputDims).setByCoord(c -> {
            int[] coords = c.getCoords();
            return Math.exp(inputTensor.get(c) - maxTensor.get(coords[0], coords[1], 0));
          });
        }).toArray(i -> new Tensor[i]));
    TensorArray sums = new TensorArray(exps.stream().map(expTensor -> {
      return new Tensor(width, height, 1).setByCoord(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(band -> {
          int[] coords = c.getCoords();
          return expTensor.get(coords[0], coords[1], band);
        }).sum();
      });
    }).toArray(i -> new Tensor[i]));
    TensorArray output = new TensorArray(
        RefIntStream.range(0, inputData.length()).mapToObj(index -> {
          Tensor sumTensor = sums.get(index);
          Tensor expTensor = exps.get(index);
          return new Tensor(inputDims).setByCoord(c -> {
            int[] coords = c.getCoords();
            return (expTensor.get(c) / sumTensor.get(coords[0], coords[1], 0));
          });
        }).toArray(i -> new Tensor[i]));
    return new Result(output, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {

        TensorArray dots = new TensorArray(
            RefIntStream.range(0, inputData.length()).mapToObj(index -> {
              final Tensor deltaTensor = delta.get(index);
              Tensor expTensor = exps.get(index);
              return new Tensor(width, height, 1).setByCoord(c -> {
                return RefIntStream.range(0, inputBands).mapToDouble(band -> {
                  int[] coords = c.getCoords();
                  return expTensor.get(coords[0], coords[1], band) * deltaTensor.get(coords[0], coords[1], band);
                }).sum();
              });
            }).toArray(i -> new Tensor[i]));

        TensorArray passback = new TensorArray(
            RefIntStream.range(0, inputData.length()).mapToObj(index -> {
              final Tensor deltaTensor = delta.get(index);
              final Tensor expTensor = exps.get(index);
              Tensor sumTensor = sums.get(index);
              Tensor dotTensor = dots.get(index);
              return new Tensor(inputDims).setByCoord(c -> {
                int[] coords = c.getCoords();
                double sum = sumTensor.get(coords[0], coords[1], 0);
                double dot = dotTensor.get(coords[0], coords[1], 0);
                double deltaValue = deltaTensor.get(c);
                double expValue = expTensor.get(c);
                return (sum * deltaValue - dot) * expValue / (sum * sum);
              });
            }).toArray(i -> new Tensor[i]));

        input.accumulate(buffer, passback);
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
  ImgPixelSoftmaxLayer addRef() {
    return (ImgPixelSoftmaxLayer) super.addRef();
  }
}
