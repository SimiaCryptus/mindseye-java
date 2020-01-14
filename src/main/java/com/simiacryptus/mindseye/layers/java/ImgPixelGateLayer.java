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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
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

  @Nullable
  public static @SuppressWarnings("unused")
  ImgPixelGateLayer[] addRefs(@Nullable ImgPixelGateLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRef)
        .toArray((x) -> new ImgPixelGateLayer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  ImgPixelGateLayer[][] addRefs(@Nullable ImgPixelGateLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRefs)
        .toArray((x) -> new ImgPixelGateLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 2 == inObj.length;
    Result temp_29_0006 = eval(inObj[0].addRef(), inObj[1].addRef());
    ReferenceCounting.freeRefs(inObj);
    return temp_29_0006;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input, @Nonnull final Result gate) {
    final TensorList inputData = input.getData();
    final TensorList gateData = gate.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    try {
      try {
        try {
          try {
            return new Result(new TensorArray(RefIntStream.range(0, inputData.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                  Tensor inputTensor = inputData.get(i);
                  Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
                  Tensor temp_29_0009 = new Tensor(inputDims[0], inputDims[1], inputDims[2]);
                  Tensor temp_29_0003 = temp_29_0009
                      .setByCoord(RefUtil.wrapInterface((ToDoubleFunction<Coordinate>) c -> {
                            int[] coords = c.getCoords();
                            return inputTensor.get(coords[0], coords[1], coords[2])
                                * gateTensor.get(coords[0], coords[1], 0);
                          }, inputTensor.addRef(),
                          gateTensor.addRef()));
                  temp_29_0009.freeRef();
                  gateTensor.freeRef();
                  inputTensor.freeRef();
                  return temp_29_0003;
                }, inputData.addRef(), gateData.addRef()))
                .toArray(i -> new Tensor[i])), new Result.Accumulator() {
              {
                input.addRef();
                gate.addRef();
                gateData.addRef();
                inputData.addRef();
              }

              @Override
              public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
                if (input.isAlive()) {
                  @Nonnull
                  TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                      .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                            Tensor deltaTensor = delta.get(i);
                            Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
                            TensorList temp_29_0012 = input.getData();
                            Tensor temp_29_0010 = new Tensor(temp_29_0012.getDimensions());
                            temp_29_0012.freeRef();
                            Tensor temp_29_0004 = temp_29_0010.setByCoord(RefUtil.wrapInterface(c -> {
                                  int[] coords = c.getCoords();
                                  return deltaTensor.get(coords[0], coords[1], coords[2])
                                      * gateTensor.get(coords[0], coords[1], 0);
                                }, gateTensor.addRef(),
                                deltaTensor.addRef()));
                            temp_29_0010.freeRef();
                            gateTensor.freeRef();
                            deltaTensor.freeRef();
                            return temp_29_0004;
                          }, delta.addRef(), input.addRef(),
                          gateData.addRef()))
                      .toArray(i -> new Tensor[i]));
                  input.accumulate(buffer == null ? null : buffer.addRef(),
                      tensorArray);
                }
                if (gate.isAlive()) {
                  @Nonnull
                  TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                      .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                            Tensor deltaTensor = delta.get(i);
                            Tensor inputTensor = inputData.get(i);
                            Tensor temp_29_0011 = new Tensor(gateData.getDimensions());
                            Tensor temp_29_0005 = temp_29_0011.setByCoord(RefUtil.wrapInterface(
                                c -> RefIntStream.range(0, inputDims[2]).mapToDouble(RefUtil.wrapInterface(b -> {
                                      int[] coords = c.getCoords();
                                      return deltaTensor.get(coords[0], coords[1], b)
                                          * inputTensor.get(coords[0], coords[1], b);
                                    }, inputTensor.addRef(),
                                    deltaTensor.addRef())).sum(),
                                inputTensor.addRef(),
                                deltaTensor.addRef()));
                            temp_29_0011.freeRef();
                            inputTensor.freeRef();
                            deltaTensor.freeRef();
                            return temp_29_0005;
                          }, gateData.addRef(), delta.addRef(),
                          inputData.addRef()))
                      .toArray(i -> new Tensor[i]));
                  gate.accumulate(buffer == null ? null : buffer.addRef(),
                      tensorArray);
                }
                delta.freeRef();
                if (null != buffer)
                  buffer.freeRef();
              }

              public @SuppressWarnings("unused")
              void _free() {
                gate.freeRef();
                input.freeRef();
                gateData.freeRef();
                inputData.freeRef();
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
          } finally {
            gate.freeRef();
          }
        } finally {
          input.freeRef();
        }
      } finally {
        gateData.freeRef();
      }
    } finally {
      inputData.freeRef();
    }
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
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPixelGateLayer addRef() {
    return (ImgPixelGateLayer) super.addRef();
  }
}
