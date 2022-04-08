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

import static com.simiacryptus.mindseye.lang.Result.anyAlive;

/**
 * This class is responsible for reducing the values in a dataset by summing them up.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class SumReducerLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);

  /**
   * Instantiates a new Sum reducer layer.
   */
  public SumReducerLayer() {
  }

  /**
   * Instantiates a new Sum reducer layer.
   *
   * @param id the id
   */
  protected SumReducerLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * Creates a new {@link SumReducerLayer} from a JSON object.
   *
   * @param json the JSON object to use for creating the layer
   * @param rs   a map of character sequences to byte arrays
   * @return a new {@link SumReducerLayer}
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SumReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumReducerLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_62_0003 = inObj[0].getData();
    int length = temp_62_0003.length();
    temp_62_0003.freeRef();
    boolean alive = anyAlive(RefUtil.addRef(inObj));
    TensorArray data = fwd(RefUtil.addRef(inObj), length);
    Accumulator accumulator = new Accumulator(inObj);
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
  SumReducerLayer addRef() {
    return (SumReducerLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(@Nonnull Result[] inObj, int length) {
    return new TensorArray(RefIntStream.range(0, length).parallel()
        .mapToDouble(RefUtil.wrapInterface(dataIndex -> {
          double sum = 0;
          for (@Nonnull final Result element : inObj) {
            TensorList data = element.getData();
            @Nullable
            Tensor tensor = data.get(dataIndex);
            data.freeRef();
            @Nullable final double[] input = tensor.getData();
            tensor.freeRef();
            for (final double element2 : input) {
              sum += element2;
            }
          }
          return sum;
        }, inObj)).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1}))
        .toArray(Tensor[]::new));
  }

  /**
   * The Accumulator class is used to store an array of Result objects.
   *
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final Result[] inObj;

    /**
     * Instantiates a new Accumulator.
     *
     * @param inObj the in obj
     */
    public Accumulator(Result... inObj) {
      this.inObj = inObj;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      for (@Nonnull final Result in_l : inObj) {
        if (in_l.isAlive()) {
          TensorList data1 = in_l.getData();
          @Nonnull
          TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data1.length()).parallel()
              .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
                Tensor tensor = data.get(dataIndex);
                assert 1 == tensor.length() : RefArrays.toString(tensor.getDimensions());
                @Nonnull final Tensor passback = new Tensor(data1.getDimensions());
                for (int i = 0; i < Tensor.length(data1.getDimensions()); i++) {
                  passback.set(i, tensor.get(0));
                }
                tensor.freeRef();
                return passback;
              }, data.addRef(), in_l.addRef(), data1))
              .toArray(Tensor[]::new));
          DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          Result.Accumulator accumulator = in_l.getAccumulator();
          try {
            accumulator.accept(buffer1, tensorArray);
          } finally {
            accumulator.freeRef();
          }
        }
      }
      data.freeRef();
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
      RefUtil.freeRef(inObj);
    }
  }
}
