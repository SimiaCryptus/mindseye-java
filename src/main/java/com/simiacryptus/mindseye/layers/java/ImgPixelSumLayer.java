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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class ImgPixelSumLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSumLayer.class);

  public ImgPixelSumLayer() {
    super();
  }

  protected ImgPixelSumLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPixelSumLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSumLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result temp_47_0004 = eval(inObj[0].addRef());
    RefUtil.freeRefs(inObj);
    return temp_47_0004;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    try {
      return new Result(new TensorArray(inputData.stream().map(tensor -> {
        Tensor temp_47_0006 = new Tensor(inputDims[0], inputDims[1], 1);
        final ToDoubleFunction<Coordinate> f = RefUtil.wrapInterface((ToDoubleFunction<Coordinate>) c -> {
          return RefIntStream.range(0, inputDims[2]).mapToDouble(RefUtil.wrapInterface((IntToDoubleFunction) i -> {
            int[] coords = c.getCoords();
            return tensor.get(coords[0], coords[1], i);
          }, tensor == null ? null : tensor.addRef())).sum();
        }, tensor == null ? null : tensor.addRef());
        temp_47_0006.setByCoord(f);
        Tensor temp_47_0002 = temp_47_0006.addRef();
        temp_47_0006.freeRef();
        if (null != tensor)
          tensor.freeRef();
        return temp_47_0002;
      }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
        {
          input.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (input.isAlive()) {
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
            }).toArray(i -> new Tensor[i]));
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
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
          super._free();
        }
      };
    } finally {
      input.freeRef();
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPixelSumLayer addRef() {
    return (ImgPixelSumLayer) super.addRef();
  }
}
