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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
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

  @SuppressWarnings("unused")
  public static ImgPixelGateLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelGateLayer(json);
  }

  public static @SuppressWarnings("unused") ImgPixelGateLayer[] addRefs(ImgPixelGateLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRef)
        .toArray((x) -> new ImgPixelGateLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgPixelGateLayer[][] addRefs(ImgPixelGateLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelGateLayer::addRefs)
        .toArray((x) -> new ImgPixelGateLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 2 == inObj.length;
    Result temp_29_0006 = eval(inObj[0].addRef(), inObj[1].addRef());
    if (null != inObj)
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
                      }, inputTensor == null ? null : inputTensor.addRef(),
                          gateTensor == null ? null : gateTensor.addRef()));
                  if (null != temp_29_0009)
                    temp_29_0009.freeRef();
                  if (null != gateTensor)
                    gateTensor.freeRef();
                  if (null != inputTensor)
                    inputTensor.freeRef();
                  return temp_29_0003;
                }, inputData == null ? null : inputData.addRef(), gateData == null ? null : gateData.addRef()))
                .toArray(i -> new Tensor[i])), new Result.Accumulator() {
                  {
                    input.addRef();
                    gate.addRef();
                    gateData.addRef();
                    inputData.addRef();
                  }

                  @Override
                  public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                    if (input.isAlive()) {
                      @Nonnull
                      TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                            Tensor deltaTensor = delta.get(i);
                            Tensor gateTensor = gateData.get(gateData.length() == 1 ? 0 : i);
                            TensorList temp_29_0012 = input.getData();
                            Tensor temp_29_0010 = new Tensor(temp_29_0012.getDimensions());
                            if (null != temp_29_0012)
                              temp_29_0012.freeRef();
                            Tensor temp_29_0004 = temp_29_0010.setByCoord(RefUtil.wrapInterface(c -> {
                              int[] coords = c.getCoords();
                              return deltaTensor.get(coords[0], coords[1], coords[2])
                                  * gateTensor.get(coords[0], coords[1], 0);
                            }, gateTensor == null ? null : gateTensor.addRef(),
                                deltaTensor == null ? null : deltaTensor.addRef()));
                            if (null != temp_29_0010)
                              temp_29_0010.freeRef();
                            if (null != gateTensor)
                              gateTensor.freeRef();
                            if (null != deltaTensor)
                              deltaTensor.freeRef();
                            return temp_29_0004;
                          }, delta == null ? null : delta.addRef(), input == null ? null : input.addRef(),
                              gateData == null ? null : gateData.addRef()))
                          .toArray(i -> new Tensor[i]));
                      input.accumulate(buffer == null ? null : buffer.addRef(),
                          tensorArray == null ? null : tensorArray);
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
                                }, inputTensor == null ? null : inputTensor.addRef(),
                                    deltaTensor == null ? null : deltaTensor.addRef())).sum(),
                                inputTensor == null ? null : inputTensor.addRef(),
                                deltaTensor == null ? null : deltaTensor.addRef()));
                            if (null != temp_29_0011)
                              temp_29_0011.freeRef();
                            if (null != inputTensor)
                              inputTensor.freeRef();
                            if (null != deltaTensor)
                              deltaTensor.freeRef();
                            return temp_29_0005;
                          }, gateData == null ? null : gateData.addRef(), delta == null ? null : delta.addRef(),
                              inputData == null ? null : inputData.addRef()))
                          .toArray(i -> new Tensor[i]));
                      gate.accumulate(buffer == null ? null : buffer.addRef(),
                          tensorArray == null ? null : tensorArray);
                    }
                    if (null != delta)
                      delta.freeRef();
                    if (null != buffer)
                      buffer.freeRef();
                  }

                  public @SuppressWarnings("unused") void _free() {
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
        if (null != gateData)
          gateData.freeRef();
      }
    } finally {
      if (null != inputData)
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgPixelGateLayer addRef() {
    return (ImgPixelGateLayer) super.addRef();
  }
}
