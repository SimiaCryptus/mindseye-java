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
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.IntArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class MaxDropoutNoiseLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxDropoutNoiseLayer.class);
  @Nullable
  private final int[] kernelSize;
  private final Function<IntArray, RefList<RefList<Coordinate>>> getCellMap_cached = Util.cache(this::getCellMap);

  public MaxDropoutNoiseLayer() {
    this(2, 2);
  }

  public MaxDropoutNoiseLayer(@org.jetbrains.annotations.Nullable final int... dims) {
    super();
    kernelSize = dims;
  }

  protected MaxDropoutNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    kernelSize = JsonUtil.getIntArray(json.getAsJsonArray("kernelSize"));
  }

  @SuppressWarnings("unused")
  public static MaxDropoutNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxDropoutNoiseLayer(json);
  }

  public static @SuppressWarnings("unused") MaxDropoutNoiseLayer[] addRefs(MaxDropoutNoiseLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayer::addRef)
        .toArray((x) -> new MaxDropoutNoiseLayer[x]);
  }

  public static @SuppressWarnings("unused") MaxDropoutNoiseLayer[][] addRefs(MaxDropoutNoiseLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MaxDropoutNoiseLayer::addRefs)
        .toArray((x) -> new MaxDropoutNoiseLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    final TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    final Tensor[] mask = RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          @Nullable
          final Tensor input = data0.get(dataIndex);
          @Nullable
          final Tensor output = input.map(x -> 0);
          final RefList<RefList<Coordinate>> cells = getCellMap_cached.apply(new IntArray(output.getDimensions()));
          cells.forEach(RefUtil.wrapInterface((Consumer<? super RefList<Coordinate>>) cell -> {
            RefUtil.freeRef(output.set(RefUtil.get(cell.stream()
                .max(RefComparator.comparingDouble(RefUtil.wrapInterface(
                    (ToDoubleFunction<? super Coordinate>) c -> input.get(c), input == null ? null : input.addRef())))), 1));
            if (null != cell)
              cell.freeRef();
          }, input == null ? null : input.addRef(), output == null ? null : output.addRef()));
          if (null != cells)
            cells.freeRef();
          if (null != input)
            input.freeRef();
          return output;
        }, data0 == null ? null : data0.addRef())).toArray(i -> new Tensor[i]);
    try {
      try {
        try {
          return new Result(new TensorArray(RefIntStream.range(0, itemCnt)
              .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                Tensor inputData = data0.get(dataIndex);
                @Nullable
                final double[] input = inputData.getData();
                @Nullable
                final double[] maskT = mask[dataIndex].getData();
                @Nonnull
                final Tensor output = new Tensor(inputData.getDimensions());
                if (null != inputData)
                  inputData.freeRef();
                @Nullable
                final double[] outputData = output.getData();
                for (int i = 0; i < outputData.length; i++) {
                  outputData[i] = input[i] * maskT[i];
                }
                return output;
              }, data0 == null ? null : data0.addRef(), Tensor.addRefs(mask))).toArray(i -> new Tensor[i])),
              new Result.Accumulator() {
                {
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (in0.isAlive()) {
                    @Nonnull
                    TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                          Tensor deltaTensor = delta.get(dataIndex);
                          @Nullable
                          final double[] deltaData = deltaTensor.getData();
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          @Nonnull
                          final int[] dims = data0.getDimensions();
                          @Nullable
                          final double[] maskData = mask[dataIndex].getData();
                          @Nonnull
                          final Tensor passback = new Tensor(dims);
                          for (int i = 0; i < passback.length(); i++) {
                            RefUtil.freeRef(passback.set(i, maskData[i] * deltaData[i]));
                          }
                          return passback;
                        }, data0 == null ? null : data0.addRef(), delta == null ? null : delta.addRef(),
                            Tensor.addRefs(mask)))
                        .toArray(i -> new Tensor[i]));
                    in0.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused") void _free() {
                }
              }) {

            {
            }

            @Override
            public boolean isAlive() {
              return in0.isAlive() || !isFrozen();
            }

            public void _free() {
            }

          };
        } finally {
          if (null != mask)
            ReferenceCounting.freeRefs(mask);
        }
      } finally {
        if (null != data0)
          data0.freeRef();
      }
    } finally {
      if (null != in0)
        in0.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") MaxDropoutNoiseLayer addRef() {
    return (MaxDropoutNoiseLayer) super.addRef();
  }

  private RefList<RefList<Coordinate>> getCellMap(@Nonnull final IntArray dims) {
    Tensor tensor = new Tensor(dims.data);
    RefMap<Integer, RefList<Coordinate>> temp_42_0005 = tensor.coordStream(true)
        .collect(RefCollectors.groupingBy((@Nonnull final Coordinate c) -> {
          int cellId = 0;
          int max = 0;
          for (int dim = 0; dim < dims.size(); dim++) {
            final int pos = c.getCoords()[dim] / kernelSize[dim];
            cellId = cellId * max + pos;
            max = dims.get(dim) / kernelSize[dim];
          }
          return cellId;
        }));
    RefArrayList<RefList<Coordinate>> temp_42_0004 = new RefArrayList<>(temp_42_0005.values());
    if (null != temp_42_0005)
      temp_42_0005.freeRef();
    if (null != tensor)
      tensor.freeRef();
    return temp_42_0004;
  }

}
