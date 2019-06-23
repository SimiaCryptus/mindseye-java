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
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class MaxImageBandLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);

  public MaxImageBandLayer() {
    super();
  }

  protected MaxImageBandLayer(@Nonnull final JsonObject id, final int... kernelDims) {
    super(id);
  }

  public static MaxImageBandLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxImageBandLayer(json,
        JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {

    assert 1 == inObj.length;
    final TensorList inputData = inObj[0].getData();
    inputData.addRef();
    inputData.length();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());

    final Coordinate[][] maxCoords = inputData.stream().map(data -> {
      Coordinate[] coordinates = IntStream.range(0, inputDims[2]).mapToObj(band -> {
        return data.coordStream(true).filter(e -> e.getCoords()[2] == band).max(Comparator.comparing(c -> data.get(c))).get();
      }).toArray(i -> new Coordinate[i]);
      data.freeRef();
      return coordinates;
    }).toArray(i -> new Coordinate[i][]);

    return new Result(TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(dataIndex -> {
      Tensor tensor = inputData.get(dataIndex);
      final DoubleStream doubleStream = IntStream.range(0, inputDims[2]).mapToDouble(band -> {
        final int[] maxCoord = maxCoords[dataIndex][band].getCoords();
        double v = tensor.get(maxCoord[0], maxCoord[1], band);
        return v;
      });
      Tensor tensor1 = new Tensor(1, 1, inputDims[2]).set(Tensor.getDoubles(doubleStream, inputDims[2]));
      tensor.freeRef();
      return tensor1;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (inObj[0].isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @Nonnull final Tensor passback = new Tensor(inputData.getDimensions());
          IntStream.range(0, inputDims[2]).forEach(b -> {
            final int[] maxCoord = maxCoords[dataIndex][b].getCoords();
            passback.set(new int[]{maxCoord[0], maxCoord[1], b}, deltaTensor.get(0, 0, b));
          });
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
      }
      delta.freeRef();
    }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        inputData.freeRef();
      }


      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  public static class CalcRegionsParameter {
    public int[] inputDims;
    public int[] kernelDims;

    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }

    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      @Nonnull final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
      if (!Arrays.equals(inputDims, other.inputDims)) {
        return false;
      }
      return Arrays.equals(kernelDims, other.kernelDims);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(inputDims);
      result = prime * result + Arrays.hashCode(kernelDims);
      return result;
    }

  }
}
